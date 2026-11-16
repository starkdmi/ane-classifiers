import Accelerate
import CoreImage
import CoreML
import CoreVideo
import Foundation
import os
import QuartzCore
import Vision

enum PipelineError: LocalizedError {
    case missingModel(String)

    var errorDescription: String? {
        switch self {
        case .missingModel(let name): return "\(name).mlmodelc is missing from the app bundle"
        }
    }
}

/// Runs the pipeline on a private serial queue. Camera frames only replace the "latest frame";
/// the pipeline picks it up whenever it is idle, so latency never builds up behind the camera.
/// Uncapped mode re-runs on the latest frame as soon as a run ends (shows the pipeline's capacity
/// and keeps the Neural Engine continuously busy).
final class Engine {
    /// Called on the inference queue.
    var onResult: ((FrameResult) -> Void)?

    private struct Pending {
        var running = false
        var frame: CVPixelBuffer?
        var captureTime: CFTimeInterval = 0  // host time (CACurrentMediaTime base)
        var arrivalTime: CFTimeInterval = 0  // when the app received the frame
        var isNew = false
        var uncapped = false
        // ANE wake-up pings.
        var wakeMode: ANEWakeMode = .off
        /// Paused while another pipeline (d1) is selected: frames are ignored.
        var paused = false
        var usesANE = true
        var frameInterval: CFTimeInterval = 1.0 / 30  // smoothed camera frame interval
        var lastArrival: CFTimeInterval = 0
    }

    /// How long before the predicted next frame the "predictive" ping is sent.
    var wakeLead: CFTimeInterval = 0.008
    private let waker = ANEWaker()

    private let queue = DispatchQueue(label: "inference", qos: .userInteractive)
    private let pipeline = InferencePipeline()
    // Holds a CVPixelBuffer (not Sendable); every access goes through the lock.
    private let pending = OSAllocatedUnfairLock(uncheckedState: Pending())
    private let logger = Logger(subsystem: "com.example.ANEDemo", category: "profile")
    // Touched only on `queue`.
    private let profiler = Profiler()
    private var lastResultTime: CFTimeInterval = 0
    private var fps: Double = 0
    private var modeLabel = ""

    init() {
        pipeline.profiler = profiler
    }

    func load(mode: ComputeMode, completion: @escaping (Result<LoadInfo, Error>) -> Void) {
        queue.async { [pipeline, self] in
            let start = CACurrentMediaTime()
            modeLabel = mode.rawValue
            do {
                try pipeline.load(mode: mode)
                waker.configure(model: pipeline.wakeModel, input: pipeline.wakeInput)
                pending.withLockUnchecked { $0.usesANE = pipeline.wakeModel != nil }
                completion(.success(LoadInfo(
                    seconds: CACurrentMediaTime() - start,
                    expressionClasses: pipeline.expressionClasses,
                    gestureClasses: pipeline.gestureClasses
                )))
            } catch {
                completion(.failure(error))
            }
        }
    }

    func setFaceDetector(_ kind: FaceDetectorKind) {
        queue.async { [pipeline] in
            pipeline.faceDetector = kind
            pipeline.resetTracking()
        }
    }

    /// Stops (or resumes) taking camera frames, e.g. while d1 is the selected pipeline.
    func setPaused(_ on: Bool) {
        pending.withLockUnchecked { p in
            p.paused = on
            if on { p.frame = nil }  // also ends an uncapped loop: it has no frame to re-run
        }
    }

    func setWakeMode(_ mode: ANEWakeMode) {
        pending.withLockUnchecked { $0.wakeMode = mode }
    }

    func setUncapped(_ on: Bool) {
        let kick = pending.withLockUnchecked { p -> Bool in
            p.uncapped = on
            guard on, !p.running, p.frame != nil else { return false }
            p.running = true
            return true
        }
        if kick { queue.async { [self] in runNext() } }
    }

    /// Safe to call from any thread (the camera's video queue in practice).
    /// `captureTime` is the frame's capture time on the host clock (CACurrentMediaTime base).
    func submit(_ pixelBuffer: CVPixelBuffer, captureTime: CFTimeInterval) {
        let arrival = CACurrentMediaTime()
        let (kick, pingNow) = pending.withLockUnchecked { p -> (Bool, Bool) in
            if p.paused { return (false, false) }
            if p.lastArrival > 0 {
                let delta = arrival - p.lastArrival
                if delta > 0.004, delta < 0.2 { p.frameInterval = p.frameInterval * 0.9 + delta * 0.1 }
            }
            p.lastArrival = arrival
            p.frame = pixelBuffer
            p.captureTime = captureTime
            p.arrivalTime = arrival
            p.isNew = true
            let ping = p.wakeMode == .arrival && p.usesANE && !p.uncapped && !p.running
            if p.running { return (false, ping) }
            p.running = true
            return (true, ping)
        }
        if pingNow { waker.ping() }  // wakes the ANE while the CPU downscales this frame
        if kick { queue.async { [self] in runNext() } }
    }

    /// On `queue`. Processes the latest frame, then re-dispatches itself (instead of looping) so that
    /// model loads and setting changes queued in the meantime get their turn.
    private func runNext() {
        let job = pending.withLockUnchecked { p -> (frame: CVPixelBuffer, captureTime: CFTimeInterval, arrivalTime: CFTimeInterval, isNew: Bool)? in
            guard pipeline.isLoaded, let frame = p.frame, p.isNew || p.uncapped else {
                p.running = false
                return nil
            }
            let isNew = p.isNew
            p.isNew = false
            return (frame, p.captureTime, p.arrivalTime, isNew)
        }
        guard let job else { return }
        if job.isNew {
            if job.captureTime > 0 { profiler.record("camera→app", (job.arrivalTime - job.captureTime) * 1000) }
            profiler.record("wait", (CACurrentMediaTime() - job.arrivalTime) * 1000)
        }

        do {
            var result = try pipeline.process(job.frame)
            let now = CACurrentMediaTime()
            if lastResultTime > 0 {
                let instant = 1 / max(now - lastResultTime, 1e-3)
                fps = fps == 0 ? instant : fps * 0.9 + instant * 0.1
            }
            lastResultTime = now
            result.inferenceFPS = fps
            if job.captureTime > 0 {
                let age = (now - job.captureTime) * 1000
                profiler.record(job.isNew ? "capture→result" : "frame age", age)
                if job.isNew {
                    result.latencyMs = age
                    result.cameraDelayMs = (job.arrivalTime - job.captureTime) * 1000
                }
            }
            onResult?(result)
        } catch {
            NSLog("Inference failed: \(error.localizedDescription)")
        }

        // Predictive wake-up: ping shortly before the next camera frame is expected.
        let wakeAt = pending.withLockUnchecked { p -> CFTimeInterval? in
            guard job.isNew, p.wakeMode == .predictive, p.usesANE, !p.uncapped else { return nil }
            return job.arrivalTime + p.frameInterval - wakeLead
        }
        if let wakeAt { waker.ping(at: wakeAt) }
        if let pingMs = waker.takeLastPingMs() { profiler.record("wake.ping", pingMs) }

        if let report = profiler.reportIfDue() {
            let (uncapped, wake) = pending.withLockUnchecked { ($0.uncapped, $0.wakeMode) }
            let header = "[\(modeLabel) · \(pipeline.faceDetector.rawValue) · \(uncapped ? "uncapped" : "per camera frame") · \(wake.shortName)]"
            logger.info("\(header, privacy: .public) \(report, privacy: .public)")
        }
        queue.async { [self] in runNext() }
    }
}


/// Sends tiny Neural Engine requests (BlazeFace on a blank image, ~1 ms when awake) so the ANE is
/// already awake when a frame's first real model call arrives. Uses its own model instance and queue.
final class ANEWaker: @unchecked Sendable {
    private let queue = DispatchQueue(label: "inference.wake", qos: .userInteractive)
    private let state = OSAllocatedUnfairLock(uncheckedState: (model: MLModel?, input: MLFeatureProvider?)(nil, nil))
    private let lastPing = OSAllocatedUnfairLock<Double?>(initialState: nil)

    func configure(model: MLModel?, input: MLFeatureProvider?) {
        state.withLockUnchecked { $0 = (model, input) }
    }

    /// Pings now, or at `hostTime` (CACurrentMediaTime base) if given and still in the future.
    func ping(at hostTime: CFTimeInterval? = nil) {
        let delay = hostTime.map { max(0, $0 - CACurrentMediaTime()) } ?? 0
        queue.asyncAfter(deadline: .now() + delay) { [self] in
            let (model, input) = state.withLockUnchecked { $0 }
            guard let model, let input else { return }
            let start = CACurrentMediaTime()
            _ = try? model.prediction(from: input)
            let ms = (CACurrentMediaTime() - start) * 1000
            lastPing.withLock { $0 = ms }
        }
    }

    /// Duration of the most recent ping (≈ the wake-up cost it absorbed), once.
    func takeLastPingMs() -> Double? {
        lastPing.withLock { value in
            defer { value = nil }
            return value
        }
    }
}

/// Per frame: one vImage downscale/letterbox to the YOLO input size, which feeds the face detector
/// (BlazeFace or Vision) and YOLOv10n; the face for EmotiEffLib is cropped from the full-resolution frame.
/// Not thread-safe: used only from Engine's serial queue.
final class InferencePipeline {
    private(set) var isLoaded = false
    private(set) var expressionClasses: [String] = []
    private(set) var gestureClasses: [String] = []

    /// Which face detector feeds the expression model; switchable at runtime.
    var faceDetector: FaceDetectorKind = .blazeFace
    /// Per-step timings (CPU prep vs model call vs decode), summarized in the log.
    var profiler: Profiler?
    /// Own model instance + input for ANE wake-up pings; nil when face/expression don't use the ANE.
    private(set) var wakeModel: MLModel?
    private(set) var wakeInput: MLFeatureProvider?

    private let faceRequest = VNDetectFaceRectanglesRequest()
    private var blazeFaceModel: MLModel?
    private var blazeInputPool: PixelBufferPool?
    private let blazeMinScore: Float = 0.5     // MediaPipe's current default detection confidence
    private let blazeNMSThreshold: CGFloat = 0.3
    private var emotionModel: MLModel?
    private var gestureModel: MLModel?
    private var letterboxer: Letterboxer?
    private var faceCropper: FaceCropper?
    private let stabilizer = Stabilizer()
    private let visionQueue = DispatchQueue(label: "inference.vision", qos: .userInitiated)
    private let gestureQueue = DispatchQueue(label: "inference.gesture", qos: .userInteractive)
    /// Hybrid mode: the gesture model runs on the GPU concurrently with face + expression on the ANE.
    private var parallelGestures = false
    private let tileRenderer = TileRenderer(side: 384)

    private let scoreThreshold: Float = 0.3
    private let iouThreshold: CGFloat = 0.5
    private let maxHands = 4
    private let faceMargin: CGFloat = 0.08

    func load(mode: ComputeMode) throws {
        isLoaded = false
        let config = MLModelConfiguration()
        config.computeUnits = mode.units
        let gestureConfig = MLModelConfiguration()
        gestureConfig.computeUnits = mode.gestureUnits  // differs only in Hybrid (GPU)
        parallelGestures = mode == .hybrid

        let emotion = try MLModel(contentsOf: try Self.modelURL("EmotionNet"), configuration: config)
        let gesture = try MLModel(contentsOf: try Self.modelURL("GestureNet"), configuration: gestureConfig)
        emotionModel = emotion
        gestureModel = gesture
        expressionClasses = Self.classes(of: emotion)
        gestureClasses = Self.classes(of: gesture)

        let g = gesture.modelDescription.inputDescriptionsByName["image"]?.imageConstraint
        letterboxer = Letterboxer(width: g?.pixelsWide ?? 640, height: g?.pixelsHigh ?? 384)
        let e = emotion.modelDescription.inputDescriptionsByName["image"]?.imageConstraint
        faceCropper = FaceCropper(width: e?.pixelsWide ?? 224, height: e?.pixelsHigh ?? 224)

        let blaze = try MLModel(contentsOf: try Self.modelURL("BlazeFace"), configuration: config)
        let b = blaze.modelDescription.inputDescriptionsByName["image"]?.imageConstraint
        blazeFaceModel = blaze
        blazeInputPool = PixelBufferPool(width: b?.pixelsWide ?? 256, height: b?.pixelsHigh ?? 256)

        // Separate BlazeFace instance + blank input for ANE wake-up pings (only when the ANE is in play).
        wakeModel = nil
        wakeInput = nil
        if config.computeUnits == .cpuAndNeuralEngine || config.computeUnits == .all,
           let buffer = blazeInputPool?.make() {
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                memset(base, 0, CVPixelBufferGetBytesPerRow(buffer) * CVPixelBufferGetHeight(buffer))
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            let wake = try MLModel(contentsOf: try Self.modelURL("BlazeFace"), configuration: config)
            let input = try Self.imageInput(buffer)
            _ = try? wake.prediction(from: input)
            wakeModel = wake
            wakeInput = input
        }
        Self.setComputeDevice(faceRequest, mode: mode)
        stabilizer.reset()

        try warmUp()
        isLoaded = true
    }

    func process(_ pixelBuffer: CVPixelBuffer) throws -> FrameResult {
        var timings = StageTimings()
        let frameSize = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        let empty = FrameResult(frameSize: frameSize, face: nil, hands: [], faceCrop: nil, handTiles: [],
                                timings: timings, inferenceFPS: nil)
        guard let letterboxer, let faceCropper, let emotionModel, let gestureModel else { return empty }
        let t0 = CACurrentMediaTime()

        // 1. Downscale + letterbox once (CPU, vImage).
        guard let boxed = letterboxer.letterbox(pixelBuffer) else { return empty }
        let (small, transform) = boxed
        let t1 = CACurrentMediaTime()

        // Gestures: YOLOv10n on the small frame + decode of its one-to-one head. Runs after face detection
        // in the single-device modes; in Hybrid it starts first, on the GPU and its own queue, in parallel
        // with face + expression on the ANE.
        var hands: [HandResult] = []
        var gestureModelMs = 0.0, gestureDecodeMs = 0.0
        let runGestures = { [self] in
            let start = CACurrentMediaTime()
            let output = try? gestureModel.prediction(from: Self.imageInput(small))
            let modelEnd = CACurrentMediaTime()
            hands = output?.featureValue(for: "detections")?.multiArrayValue.map { decodeHands($0, transform) } ?? []
            gestureModelMs = (modelEnd - start) * 1000
            gestureDecodeMs = (CACurrentMediaTime() - modelEnd) * 1000
        }
        let gestureDone = DispatchGroup()
        if parallelGestures {
            gestureDone.enter()
            gestureQueue.async {
                runGestures()
                gestureDone.leave()
            }
        }

        // 2. Face boxes (frame-normalized, top-left).
        //    BlazeFace: ~1 ms on the ANE, so it simply runs inline.
        //    Vision: ~15-20 ms on M1 Pro whatever the device, so it runs on its own queue while
        //    YOLOv10n handles gestures here; only the expression step has to wait for it.
        let faceDone = DispatchGroup()
        var faceRects: [CGRect] = []
        var faceError: Error?
        var faceMs = 0.0
        switch faceDetector {
        case .blazeFace:
            let start = CACurrentMediaTime()
            faceRects = detectFacesBlaze(small, transform)
            faceMs = (CACurrentMediaTime() - start) * 1000
        case .vision:
            faceDone.enter()
            visionQueue.async { [faceRequest] in
                let start = CACurrentMediaTime()
                do {
                    try VNImageRequestHandler(cvPixelBuffer: small, orientation: .up).perform([faceRequest])
                    faceRects = (faceRequest.results ?? []).map { transform.normalizedRect(visionBox: $0.boundingBox) }
                } catch {
                    faceError = error
                }
                faceMs = (CACurrentMediaTime() - start) * 1000
                faceDone.leave()
            }
        }

        // 3. Gestures (sequential modes).
        if !parallelGestures { runGestures() }

        faceDone.wait()
        if let faceError { throw faceError }
        if faceDetector == .vision { profiler?.record("face.vision", faceMs) }
        let t2 = CACurrentMediaTime()

        // 4. Expression: largest face, cropped from the full-res frame, scaled to 224x224 (no aspect
        //    preservation, matching EmotiEffLib's Resize((224, 224))).
        var face: FaceResult?
        if let rect = faceRects.max(by: { $0.area < $1.area }) {
            let cropRect = rect.insetBy(dx: -rect.width * faceMargin, dy: -rect.height * faceMargin)
            let crop = profiler.map { p in p.measure("expr.prep") { faceCropper.crop(pixelBuffer, normalized: cropRect) } }
                ?? faceCropper.crop(pixelBuffer, normalized: cropRect)
            if let input = crop {
                let modelStart = CACurrentMediaTime()
                let output = try emotionModel.prediction(from: Self.imageInput(input))
                profiler?.record("expr.model", (CACurrentMediaTime() - modelStart) * 1000)
                if let probs = output.featureValue(for: "probs")?.multiArrayValue?.lastAxisFloats(),
                   let va = output.featureValue(for: "va")?.multiArrayValue?.lastAxisFloats() {
                    face = FaceResult(rect: rect, probs: probs, valence: va.first ?? 0,
                                      arousal: va.count > 1 ? va[1] : 0, topIndex: ScoreMath.argmax(probs))
                }
            }
        }
        let t3 = CACurrentMediaTime()
        gestureDone.wait()  // Hybrid: join the GPU gesture branch
        let t4 = CACurrentMediaTime()
        profiler?.record("gesture.model", gestureModelMs)
        profiler?.record("gesture.decode", gestureDecodeMs)
        if parallelGestures { profiler?.record("join wait", (t4 - t3) * 1000) }

        timings.letterbox = (t1 - t0) * 1000
        timings.faceDetect = faceMs
        timings.gesture = gestureModelMs + gestureDecodeMs
        timings.expression = (t3 - t2) * 1000
        timings.total = (t4 - t0) * 1000  // wall time: Vision face detection and Hybrid gestures run in parallel
        profiler?.record("letterbox", timings.letterbox)
        profiler?.record("total", timings.total)

        // Steady boxes / scores / labels for display, then zoom tiles from the steady boxes
        // (neither counted in the timings).
        let (stableFace, stableHands) = stabilizer.update(face: face, hands: hands, time: t4)
        let tileHands = stableHands.prefix(2).sorted { $0.rect.midX < $1.rect.midX }  // left-to-right on screen
        let result = FrameResult(
            frameSize: frameSize,
            face: stableFace,
            hands: stableHands,
            faceCrop: stableFace.flatMap { crop(pixelBuffer, around: $0.rect, zoom: 1.6) },
            handTiles: tileHands.map { HandTile(hand: $0, image: crop(pixelBuffer, around: $0.rect, zoom: 1.4)) },
            timings: timings,
            inferenceFPS: nil
        )
        profiler?.record("tiles (untimed)", (CACurrentMediaTime() - t4) * 1000)
        return result
    }

    func resetTracking() { stabilizer.reset() }

    // MARK: - Helpers

    /// BlazeFace (MediaPipe "back" model, 256x256) on a square, black-barred letterbox of the small
    /// frame's content. Returns frame-normalized, top-left rects after MediaPipe's weighted NMS.
    private func detectFacesBlaze(_ small: CVPixelBuffer, _ transform: LetterboxTransform) -> [CGRect] {
        guard let model = blazeFaceModel, let pool = blazeInputPool, let input = pool.make() else { return [] }
        let contentW = Int((transform.sourceWidth * transform.scale).rounded())
        let contentH = Int((transform.sourceHeight * transform.scale).rounded())
        let fit = min(Double(pool.width) / Double(contentW), Double(pool.height) / Double(contentH))
        let dw = Int((Double(contentW) * fit).rounded()), dh = Int((Double(contentH) * fit).rounded())
        let ox = (pool.width - dw) / 2, oy = (pool.height - dh) / 2
        let prepStart = CACurrentMediaTime()
        guard PixelOps.scale(small, from: PixelRect(x: Int(transform.padX), y: Int(transform.padY), width: contentW, height: contentH),
                             into: input, at: PixelRect(x: ox, y: oy, width: dw, height: dh), padValue: 0) else { return [] }
        let modelStart = CACurrentMediaTime()
        let prediction = try? model.prediction(from: Self.imageInput(input))
        let modelEnd = CACurrentMediaTime()
        profiler?.record("face.prep", (modelStart - prepStart) * 1000)
        profiler?.record("face.model", (modelEnd - modelStart) * 1000)
        defer { profiler?.record("face.decode", (CACurrentMediaTime() - modelEnd) * 1000) }
        guard let output = prediction,
              let boxes = output.featureValue(for: "boxes")?.multiArrayValue,
              let scores = output.featureValue(for: "scores")?.multiArrayValue,
              boxes.dataType == .float32, scores.dataType == .float32, boxes.shape.count == 3 else { return [] }

        let count = scores.shape.last?.intValue ?? 0
        let scoreStride = scores.strides.last?.intValue ?? 1
        let anchorStride = boxes.strides[1].intValue, coordStride = boxes.strides[2].intValue
        // Model coords are normalized to the square input; map back through the square letterbox.
        let side = (w: Double(pool.width), h: Double(pool.height))
        func fx(_ v: Float) -> CGFloat { CGFloat((Double(v) * side.w - Double(ox)) / Double(dw)) }
        func fy(_ v: Float) -> CGFloat { CGFloat((Double(v) * side.h - Double(oy)) / Double(dh)) }

        var candidates: [(rect: CGRect, score: Float)] = []
        scores.withUnsafeBufferPointer(ofType: Float.self) { sp in
            boxes.withUnsafeBufferPointer(ofType: Float.self) { bp in
                for i in 0..<count where sp[i * scoreStride] >= blazeMinScore {
                    let base = i * anchorStride
                    let x1 = fx(bp[base]), y1 = fy(bp[base + coordStride])
                    let x2 = fx(bp[base + 2 * coordStride]), y2 = fy(bp[base + 3 * coordStride])
                    guard x2 > x1, y2 > y1 else { continue }
                    candidates.append((CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1), sp[i * scoreStride]))
                }
            }
        }
        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
        return Self.weightedNMS(candidates, threshold: blazeNMSThreshold)
            .map { $0.rect.intersection(unit) }
            .filter { !$0.isNull && $0.width > 0.01 && $0.height > 0.01 }
    }

    /// MediaPipe-style NMS: overlapping detections are blended, weighted by score, instead of dropped.
    private static func weightedNMS(_ detections: [(rect: CGRect, score: Float)], threshold: CGFloat) -> [(rect: CGRect, score: Float)] {
        var remaining = detections.sorted { $0.score > $1.score }
        var output: [(rect: CGRect, score: Float)] = []
        while !remaining.isEmpty {
            let first = remaining.removeFirst()
            var group = [first]
            remaining.removeAll { candidate in
                guard candidate.rect.iou(first.rect) > threshold else { return false }
                group.append(candidate)
                return true
            }
            let total = group.reduce(Float(0)) { $0 + $1.score }
            guard group.count > 1, total > 0 else { output.append(first); continue }
            var x1: CGFloat = 0, y1: CGFloat = 0, x2: CGFloat = 0, y2: CGFloat = 0
            for d in group {
                let w = CGFloat(d.score)
                x1 += d.rect.minX * w; y1 += d.rect.minY * w; x2 += d.rect.maxX * w; y2 += d.rect.maxY * w
            }
            let t = CGFloat(total)
            output.append((CGRect(x: x1 / t, y: y1 / t, width: (x2 - x1) / t, height: (y2 - y1) / t), total / Float(group.count)))
        }
        return output
    }

    private func decodeHands(_ output: MLMultiArray, _ transform: LetterboxTransform) -> [HandResult] {
        let shape = output.shape.map(\.intValue)
        let strides = output.strides.map(\.intValue)
        guard shape.count == 3, output.dataType == .float32 else { return [] }
        let anchors = shape[1], classes = shape[2] - 4
        let anchorStride = strides[1], channelStride = strides[2]

        return output.withUnsafeBufferPointer(ofType: Float.self) { p -> [HandResult] in
            var candidates: [(anchor: Int, score: Float, cls: Int)] = []
            for a in 0..<anchors {
                let base = a * anchorStride
                var best: Float = 0
                var bestClass = 0
                for c in 0..<classes {
                    let v = p[base + (4 + c) * channelStride]
                    if v > best { best = v; bestClass = c }
                }
                if best >= scoreThreshold { candidates.append((a, best, bestClass)) }
            }
            candidates.sort { $0.score > $1.score }

            var kept: [HandResult] = []
            for candidate in candidates {
                let base = candidate.anchor * anchorStride
                let rect = transform.normalizedRect(
                    x1: p[base], y1: p[base + channelStride],
                    x2: p[base + 2 * channelStride], y2: p[base + 3 * channelStride]
                )
                // YOLOv10's one-to-one head is NMS-free; this only guards against rare duplicates.
                if kept.contains(where: { $0.rect.iou(rect) > iouThreshold }) { continue }
                let scores = (0..<classes).map { p[base + (4 + $0) * channelStride] }
                kept.append(HandResult(rect: rect, scores: scores, classIndex: candidate.cls, score: candidate.score))
                if kept.count == maxHands { break }
            }
            return kept
        }
    }

    /// A zoom tile for the UI; see `zoomTile`.
    private func crop(_ source: CVPixelBuffer, around rect: CGRect, zoom: CGFloat) -> CGImage? {
        zoomTile(source, around: rect, zoom: zoom, renderer: tileRenderer)
    }

    private func warmUp() throws {
        let savedProfiler = profiler
        profiler = nil  // keep warm-up out of the stats
        defer { profiler = savedProfiler }
        guard let blank = makePixelBuffer(width: 1920, height: 1080),
              let letterboxer, let faceCropper, let emotionModel, let gestureModel,
              let boxed = letterboxer.letterbox(blank) else { return }
        let small = boxed.0
        try VNImageRequestHandler(cvPixelBuffer: small, orientation: .up).perform([faceRequest])
        _ = detectFacesBlaze(small, boxed.1)
        if let face = faceCropper.crop(blank, normalized: CGRect(x: 0.4, y: 0.3, width: 0.2, height: 0.35)) {
            _ = try emotionModel.prediction(from: Self.imageInput(face))
        }
        _ = try gestureModel.prediction(from: Self.imageInput(small))
    }

    private static func imageInput(_ buffer: CVPixelBuffer) throws -> MLFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
    }

    private static func modelURL(_ name: String) throws -> URL {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw PipelineError.missingModel(name)
        }
        return url
    }

    private static func classes(of model: MLModel) -> [String] {
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        return metadata?["classes"]?.split(separator: ",").map(String.init) ?? []
    }

    private static func setComputeDevice(_ request: VNRequest, mode: ComputeMode) {
        guard let devices = try? request.supportedComputeStageDevices[.main] else { return }
        let match = devices.first { device in
            switch (mode, device) {
            case (.neuralEngine, .neuralEngine), (.hybrid, .neuralEngine), (.gpu, .gpu), (.cpu, .cpu): return true
            default: return false
            }
        }
        request.setComputeDevice(mode == .auto ? nil : match, for: .main)
    }
}

// MARK: - Pixel buffers (BGRA, vImage)

struct PixelRect {
    var x: Int
    var y: Int
    var width: Int
    var height: Int
}

/// IOSurface-backed BGRA buffers of one fixed size (IOSurface lets Core ML hand them to the ANE without copies).
final class PixelBufferPool {
    let width: Int
    let height: Int
    private var pool: CVPixelBufferPool?

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
    }

    func make() -> CVPixelBuffer? {
        guard let pool else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        return buffer
    }
}

enum PixelOps {
    /// Resamples `from` (pixels) of a BGRA source into `at` of a BGRA destination, optionally
    /// filling the whole destination with `padValue` first.
    static func scale(_ source: CVPixelBuffer, from s: PixelRect, into destination: CVPixelBuffer, at d: PixelRect,
                      padValue: Int32? = nil) -> Bool {
        guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA,
              s.width > 0, s.height > 0, d.width > 0, d.height > 0 else { return false }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let srcBase = CVPixelBufferGetBaseAddress(source),
              let dstBase = CVPixelBufferGetBaseAddress(destination) else { return false }
        let srcRow = CVPixelBufferGetBytesPerRow(source)
        let dstRow = CVPixelBufferGetBytesPerRow(destination)
        if let padValue { memset(dstBase, padValue, dstRow * CVPixelBufferGetHeight(destination)) }

        var src = vImage_Buffer(data: srcBase.advanced(by: s.y * srcRow + s.x * 4), height: vImagePixelCount(s.height),
                                width: vImagePixelCount(s.width), rowBytes: srcRow)
        var dst = vImage_Buffer(data: dstBase.advanced(by: d.y * dstRow + d.x * 4), height: vImagePixelCount(d.height),
                                width: vImagePixelCount(d.width), rowBytes: dstRow)
        return vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageNoFlags)) == kvImageNoError
    }
}

struct LetterboxTransform {
    var scale: Double
    var padX: Double
    var padY: Double
    var sourceWidth: Double
    var sourceHeight: Double
    var inputWidth: Double
    var inputHeight: Double

    /// Model-input pixel corners -> normalized top-left rect in the source frame.
    func normalizedRect(x1: Float, y1: Float, x2: Float, y2: Float) -> CGRect {
        func nx(_ v: Float) -> Double { min(max((Double(v) - padX) / scale / sourceWidth, 0), 1) }
        func ny(_ v: Float) -> Double { min(max((Double(v) - padY) / scale / sourceHeight, 0), 1) }
        return CGRect(x: nx(x1), y: ny(y1), width: nx(x2) - nx(x1), height: ny(y2) - ny(y1))
    }

    /// Vision's normalized, bottom-left rect on the letterboxed input -> normalized top-left rect in the source.
    func normalizedRect(visionBox b: CGRect) -> CGRect {
        normalizedRect(x1: Float(b.minX * inputWidth), y1: Float((1 - b.maxY) * inputHeight),
                       x2: Float(b.maxX * inputWidth), y2: Float((1 - b.minY) * inputHeight))
    }
}

/// Scales a frame into the YOLO input size, keeping aspect ratio and padding with gray 114
/// (Ultralytics' letterbox color).
final class Letterboxer {
    let width: Int
    let height: Int
    private let pool: PixelBufferPool

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        pool = PixelBufferPool(width: width, height: height)
    }

    func letterbox(_ source: CVPixelBuffer) -> (CVPixelBuffer, LetterboxTransform)? {
        guard let destination = pool.make() else { return nil }
        let sw = CVPixelBufferGetWidth(source), sh = CVPixelBufferGetHeight(source)
        let scale = min(Double(width) / Double(sw), Double(height) / Double(sh))
        let nw = min(width, Int((Double(sw) * scale).rounded()))
        let nh = min(height, Int((Double(sh) * scale).rounded()))
        let padX = (width - nw) / 2, padY = (height - nh) / 2
        guard PixelOps.scale(source, from: PixelRect(x: 0, y: 0, width: sw, height: sh), into: destination,
                             at: PixelRect(x: padX, y: padY, width: nw, height: nh), padValue: 114) else { return nil }
        let transform = LetterboxTransform(scale: scale, padX: Double(padX), padY: Double(padY),
                                           sourceWidth: Double(sw), sourceHeight: Double(sh),
                                           inputWidth: Double(width), inputHeight: Double(height))
        return (destination, transform)
    }
}

/// Square zoom tile around `rect` (normalized, top-left), scaled on the CPU with vImage (~0.5 ms, keeping the
/// GPU free) into a fixed-size CGImage. The square is shifted to stay inside the frame rather than clipped,
/// so tiles near the edges don't distort.
func zoomTile(_ source: CVPixelBuffer, around rect: CGRect, zoom: CGFloat, renderer: TileRenderer) -> CGImage? {
    let w = CGFloat(CVPixelBufferGetWidth(source)), h = CGFloat(CVPixelBufferGetHeight(source))
    let side = min(max(rect.width * w, rect.height * h) * zoom, w, h)
    guard side >= 8 else { return nil }
    let x = min(max(rect.midX * w - side / 2, 0), w - side)
    let y = min(max(rect.midY * h - side / 2, 0), h - side)
    return renderer.render(source, region: PixelRect(x: Int(x), y: Int(y), width: Int(side), height: Int(side)))
}

/// Renders a square region of a BGRA frame into a fixed-size CGImage on the CPU (for the zoom tiles).
final class TileRenderer {
    let side: Int
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

    init(side: Int) {
        self.side = side
    }

    func render(_ source: CVPixelBuffer, region: PixelRect) -> CGImage? {
        guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA,
              region.width > 0, region.height > 0,
              let context = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
              let destination = context.data else { return nil }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(source) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(source)
        var src = vImage_Buffer(data: base.advanced(by: region.y * rowBytes + region.x * 4),
                                height: vImagePixelCount(region.height), width: vImagePixelCount(region.width), rowBytes: rowBytes)
        var dst = vImage_Buffer(data: destination, height: vImagePixelCount(side), width: vImagePixelCount(side),
                                rowBytes: context.bytesPerRow)
        guard vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageNoFlags)) == kvImageNoError else { return nil }
        return context.makeImage()
    }
}

/// Crops a normalized region of a frame and scales it (stretching) to the classifier's input size.
final class FaceCropper {
    private let pool: PixelBufferPool

    init(width: Int, height: Int) {
        pool = PixelBufferPool(width: width, height: height)
    }

    func crop(_ source: CVPixelBuffer, normalized r: CGRect) -> CVPixelBuffer? {
        let sw = CVPixelBufferGetWidth(source), sh = CVPixelBufferGetHeight(source)
        let x0 = max(0, Int((r.minX * CGFloat(sw)).rounded(.down)))
        let y0 = max(0, Int((r.minY * CGFloat(sh)).rounded(.down)))
        let x1 = min(sw, Int((r.maxX * CGFloat(sw)).rounded(.up)))
        let y1 = min(sh, Int((r.maxY * CGFloat(sh)).rounded(.up)))
        guard x1 - x0 >= 8, y1 - y0 >= 8, let destination = pool.make() else { return nil }
        let ok = PixelOps.scale(source, from: PixelRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0), into: destination,
                                at: PixelRect(x: 0, y: 0, width: pool.width, height: pool.height))
        return ok ? destination : nil
    }
}

private func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    var buffer: CVPixelBuffer?
    let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
    CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
    if let buffer {
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            memset(base, 128, CVPixelBufferGetBytesPerRow(buffer) * height)
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
    }
    return buffer
}

// MARK: - Small geometry / array helpers

extension CGRect {
    var area: CGFloat { width * height }

    func iou(_ other: CGRect) -> CGFloat {
        let inter = intersection(other)
        guard !inter.isNull else { return 0 }
        let union = area + other.area - inter.area
        return union > 0 ? inter.area / union : 0
    }
}

extension MLMultiArray {
    /// Values along the last axis of a (1, N) array, respecting strides.
    func lastAxisFloats() -> [Float] {
        guard let n = shape.last?.intValue, let stride = strides.last?.intValue else { return [] }
        if dataType == .float32 {
            return withUnsafeBufferPointer(ofType: Float.self) { p in (0..<n).map { p[$0 * stride] } }
        }
        return (0..<n).map { self[$0].floatValue }  // logical (row-major) index
    }
}
