import CoreGraphics
import CoreVideo
import Foundation
import MLX
import os
import QuartzCore

/// One d1 decision, mapped for the shared UI.
struct D1Result {
    var decision: D1Decision
    var frameSize: CGSize
    var faceCrop: CGImage?
    var handCrop: CGImage?
    var totalMs: Double
    var visionMs: Double
    var backboneMs: Double
    var tokens: Int
    /// Capture timestamp -> decision ready, and capture -> frame arrived in the app.
    var latencyMs: Double?
    var cameraDelayMs: Double?
}

/// Runs d1-2B (MLX, GPU) as the app's pipeline when selected: newest camera frame in, one System One
/// decision (hand + face choices and boxes) out, back to back. All MLX work happens on one serial queue.
final class D1Service: @unchecked Sendable {
    enum State {
        case off
        case loading
        case ready(seconds: Double)
        case failed(String)
    }

    /// Called on the d1 queue.
    var onState: ((State) -> Void)?
    var onResult: ((D1Result) -> Void)?

    static let modelFolderName = "d1-2B-MLX-8bit"
    static let modelPathKey = "d1ModelPath"

    /// Where the d1 weights are, first match wins:
    /// 1. a folder chosen in Pipeline › Locate d1 Weights… (or `--args -d1ModelPath /path`),
    /// 2. `weights/d1-2B-MLX-8bit` in a folder above the app (building inside the repo after `tools/setup.sh --d1`),
    /// 3. `~/Library/Application Support/ANE Classifiers/d1-2B-MLX-8bit`.
    static func locateModelDirectory() -> URL? {
        func valid(_ url: URL) -> Bool {
            FileManager.default.fileExists(atPath: url.appending(path: "config.json").path)
        }
        if let path = UserDefaults.standard.string(forKey: modelPathKey) {
            let url = URL(fileURLWithPath: path)
            if valid(url) { return url }
        }
        var folder = Bundle.main.bundleURL.deletingLastPathComponent()
        for _ in 0 ..< 8 {
            let candidate = folder.appending(path: "weights/\(modelFolderName)")
            if valid(candidate) { return candidate }
            folder = folder.deletingLastPathComponent()
        }
        if let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let candidate = support.appending(path: "ANE Classifiers/\(modelFolderName)")
            if valid(candidate) { return candidate }
        }
        return nil
    }

    private struct Pending {
        var frame: CVPixelBuffer?
        var captureTime: CFTimeInterval = 0
        var arrivalTime: CFTimeInterval = 0
        var running = false
        var enabled = false
    }

    private let queue = DispatchQueue(label: "d1", qos: .userInitiated)
    private let pending = OSAllocatedUnfairLock(uncheckedState: Pending())
    // Touched only on `queue`.
    private var runner: D1Runner?
    private var scaledPool: PixelBufferPool?
    private let tileRenderer = TileRenderer(side: 384)

    func setEnabled(_ on: Bool) {
        pending.withLockUnchecked { $0.enabled = on }
        guard on else {
            queue.async { [self] in
                runner = nil  // frees the ~2.2 GB of weights
                Memory.clearCache()
                onState?(.off)
            }
            return
        }
        queue.async { [self] in
            guard runner == nil else { return }
            onState?(.loading)
            let start = CACurrentMediaTime()
            do {
                guard let planURL = Bundle.main.url(forResource: "d1_plan", withExtension: "json") else {
                    throw NSError(domain: "d1", code: 1, userInfo: [NSLocalizedDescriptionKey: "d1_plan.json missing from the app"])
                }
                let plan = try JSONDecoder().decode(D1Plan.self, from: Data(contentsOf: planURL))
                guard let directory = Self.locateModelDirectory() else {
                    throw NSError(domain: "d1", code: 2, userInfo: [NSLocalizedDescriptionKey:
                        "weights not found: run tools/setup.sh --d1, or use Pipeline › Locate d1 Weights…"])
                }
                let runner = try D1Runner(modelDirectory: directory, plan: plan)
                scaledPool = PixelBufferPool(width: plan.image.width, height: plan.image.height)
                // warm-up: builds the GPU kernels for these shapes
                _ = runner.decide(pixelValues: runner.preprocess(
                    rgb: MLXArray.zeros([plan.image.height, plan.image.width, 3], type: UInt8.self)))
                self.runner = runner
                onState?(.ready(seconds: CACurrentMediaTime() - start))
                kick()
            } catch {
                onState?(.failed(error.localizedDescription))
            }
        }
    }

    /// Any thread: keeps only the newest frame; starts a decision if idle.
    func submit(_ pixelBuffer: CVPixelBuffer, captureTime: CFTimeInterval) {
        let arrival = CACurrentMediaTime()
        let start = pending.withLockUnchecked { p -> Bool in
            guard p.enabled else { return false }
            p.frame = pixelBuffer
            p.captureTime = captureTime
            p.arrivalTime = arrival
            if p.running { return false }
            p.running = true
            return true
        }
        if start { queue.async { [self] in runNext() } }
    }

    private func kick() {
        let start = pending.withLockUnchecked { p -> Bool in
            guard p.enabled, !p.running, p.frame != nil else { return false }
            p.running = true
            return true
        }
        if start { queue.async { [self] in runNext() } }
    }

    private func runNext() {
        let job = pending.withLockUnchecked { p -> (CVPixelBuffer, CFTimeInterval, CFTimeInterval)? in
            guard runner != nil, p.enabled, let frame = p.frame else {
                p.running = false
                return nil
            }
            p.frame = nil  // each frame is decided once
            return (frame, p.captureTime, p.arrivalTime)
        }
        guard let job, let runner, let rgb = rgbArray(job.0) else { return }
        let (frame, captureTime, arrivalTime) = job

        let start = CACurrentMediaTime()
        let decision = runner.decide(pixelValues: runner.preprocess(rgb: rgb))
        let end = CACurrentMediaTime()

        // d1 saw the whole frame (stretched to its input size), so its box fractions are frame fractions.
        let faceRect = decision.boxes.first { $0.name == "face_box" }?.rect
        let handRect = decision.boxes.first { $0.name == "hand_box" }?.rect
        onResult?(D1Result(
            decision: decision,
            frameSize: CGSize(width: CVPixelBufferGetWidth(frame), height: CVPixelBufferGetHeight(frame)),
            faceCrop: faceRect.flatMap { zoomTile(frame, around: $0, zoom: 1.6, renderer: tileRenderer) },
            handCrop: handRect.flatMap { zoomTile(frame, around: $0, zoom: 1.4, renderer: tileRenderer) },
            totalMs: (end - start) * 1000,
            visionMs: runner.lastVisionMs,
            backboneMs: runner.lastBackboneMs,
            tokens: runner.tokenCount,
            latencyMs: captureTime > 0 ? (end - captureTime) * 1000 : nil,
            cameraDelayMs: captureTime > 0 ? (arrivalTime - captureTime) * 1000 : nil
        ))
        queue.async { [self] in runNext() }
    }

    /// BGRA camera frame -> scaled to the plan's size -> RGB UInt8 [H, W, 3].
    private func rgbArray(_ frame: CVPixelBuffer) -> MLXArray? {
        guard let pool = scaledPool, let scaled = pool.make() else { return nil }
        let (w, h) = (CVPixelBufferGetWidth(frame), CVPixelBufferGetHeight(frame))
        guard PixelOps.scale(frame, from: PixelRect(x: 0, y: 0, width: w, height: h), into: scaled,
                             at: PixelRect(x: 0, y: 0, width: pool.width, height: pool.height)) else { return nil }
        CVPixelBufferLockBaseAddress(scaled, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(scaled, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(scaled) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(scaled)
        var bytes = [UInt8](repeating: 0, count: pool.width * pool.height * 4)
        bytes.withUnsafeMutableBytes { dst in
            for row in 0 ..< pool.height {
                memcpy(dst.baseAddress! + row * pool.width * 4, base + row * rowBytes, pool.width * 4)
            }
        }
        let bgra = MLXArray(bytes, [pool.height, pool.width, 4])
        return bgra.take(MLXArray([Int32(2), 1, 0]), axis: 2)  // BGRA -> RGB
    }
}
