import AppKit
import AVFoundation
import Observation
import os
import SwiftUI

/// UI state. Mutated only on the main thread (camera and inference callbacks hop to main).
@Observable
final class AppModel {
    let displayLayer: AVSampleBufferDisplayLayer

    var status = "Starting…"
    var frameSize = CGSize(width: 1280, height: 720)

    var face: FaceResult?
    var hands: [HandResult] = []
    var faceCrop: CGImage?
    var handTiles: [HandTile] = []

    var expressionClasses: [String] = []
    var gestureClasses: [String] = []
    /// Display values: smoothed per track by the Stabilizer, faded out here when the face/hand is gone.
    var expressionScores: [Float] = []
    var valence: Float = 0
    var arousal: Float = 0
    var gestureScores: [Float] = []
    /// Gesture rows shown in the list (indices into gestureClasses), in a sticky order.
    var gestureRows: [Int] = []
    let gestureRowCount = 9

    var timings = StageTimings()
    var inferenceFPS = 0.0
    var cameraFPS = 0.0
    /// Camera capture -> decision ready (smoothed), for fresh frames only.
    var latencyMs = 0.0
    /// Camera capture -> frame delivered to the app (smoothed): the part before our pipeline starts.
    var cameraDelayMs = 0.0
    var loadSeconds = 0.0

    private(set) var computeMode: ComputeMode = .neuralEngine
    private(set) var faceDetector: FaceDetectorKind = .blazeFace
    /// Re-run on the latest frame as soon as a run finishes, instead of once per camera frame.
    private(set) var uncapped = false
    /// Early tiny ANE requests that hide the Neural Engine's wake-up cost between frames.
    private(set) var aneWake: ANEWakeMode = .off
    /// Use the camera's fastest mode (only offered when it beats 30 fps).
    private(set) var highFrameRate = false
    private(set) var cameraMaxFPS = 30.0
    var cameras: [CameraOption] = []
    private(set) var selectedCameraID = ""

    /// Which pipeline drives the UI: the Core ML classifiers, or d1-2B (MLX on the GPU). Only one runs.
    private(set) var pipeline: PipelineKind = .classic
    var d1Status = ""
    /// d1 timings (smoothed): vision tower + projector, backbone + readout, whole decision.
    var d1VisionMs = 0.0
    var d1BackboneMs = 0.0
    var d1TotalMs = 0.0
    var d1Tokens = 0

    @ObservationIgnored private let camera = CameraController()
    @ObservationIgnored private let engine = Engine()
    @ObservationIgnored private let d1 = D1Service()
    @ObservationIgnored private var d1LastResult: CFTimeInterval = 0
    @ObservationIgnored private let renderer: AVSampleBufferVideoRenderer
    @ObservationIgnored private var started = false
    @ObservationIgnored private var lastStatsLog: CFTimeInterval = 0
    @ObservationIgnored private let logger = Logger(subsystem: "com.example.ANEDemo", category: "stats")
    // Touched only on the camera's video queue.
    @ObservationIgnored private var cameraFrameCount = 0
    @ObservationIgnored private var cameraWindowStart = CACurrentMediaTime()

    private let fade: Float = 0.8
    private let timingSmoothing = 0.15

    init() {
        let layer = AVSampleBufferDisplayLayer()
        layer.videoGravity = .resizeAspect
        layer.backgroundColor = NSColor.black.cgColor
        displayLayer = layer
        renderer = layer.sampleBufferRenderer

        engine.onResult = { [weak self] result in
            DispatchQueue.main.async { self?.apply(result) }
        }
        camera.frameHandler = { [weak self] sampleBuffer in
            self?.handleFrame(sampleBuffer)
        }
        d1.onState = { [weak self] state in
            DispatchQueue.main.async { self?.applyD1State(state) }
        }
        d1.onResult = { [weak self] result in
            DispatchQueue.main.async { self?.applyD1(result) }
        }
    }

    func setPipeline(_ kind: PipelineKind) {
        guard kind != pipeline else { return }
        pipeline = kind
        clearResults()
        let d1On = kind == .d1
        engine.setPaused(d1On)
        d1.setEnabled(d1On)  // turning d1 off also frees its weights
        if !d1On { status = "Running on \(computeMode.rawValue)" }
    }

    /// Lets the user point at a downloaded d1-2B-MLX-8bit folder; remembered for next launches.
    func locateD1Weights() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Use Folder"
        panel.message = "Choose the d1-2B-MLX-8bit folder (it contains config.json and model.safetensors)."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard FileManager.default.fileExists(atPath: url.appending(path: "config.json").path) else {
            status = "That folder has no config.json: choose the d1-2B-MLX-8bit folder"
            return
        }
        UserDefaults.standard.set(url.path, forKey: D1Service.modelPathKey)
        if pipeline == .d1 {  // reload from the new place
            d1.setEnabled(false)
            d1.setEnabled(true)
        }
    }

    private func clearResults() {
        face = nil
        hands = []
        faceCrop = nil
        handTiles = []
        expressionScores = []
        gestureScores = []
        gestureRows = []
        valence = 0
        arousal = 0
        timings = StageTimings()
        inferenceFPS = 0
        latencyMs = 0
        d1VisionMs = 0
        d1BackboneMs = 0
        d1TotalMs = 0
        d1LastResult = 0
    }

    private func applyD1State(_ state: D1Service.State) {
        switch state {
        case .off: d1Status = ""
        case .loading: d1Status = "Loading d1-2B (2.2 GB)…"
        case .ready(let seconds): d1Status = String(format: "Running d1-2B on GPU · load %.1f s", seconds)
        case .failed(let message): d1Status = "d1 failed: \(message)"
        }
        if pipeline == .d1, !d1Status.isEmpty { status = d1Status }
        logger.info("d1: \(self.d1Status, privacy: .public)")
    }

    /// d1's answers into the same state the classic pipeline fills: boxes, tiles, card scores, timings.
    private func applyD1(_ result: D1Result) {
        guard pipeline == .d1 else { return }
        let choices = result.decision.choices, boxes = result.decision.boxes
        frameSize = result.frameSize

        // A box is shown only when its question's answer isn't "none" (no face / no gesture), as in
        // github.com/ivanfioravanti/mlx-d1-gestures, the mlx-vlm web demo this mode is compared with.
        var showFace = false
        if let answer = choices.first(where: { $0.name == "face" }) {
            expressionClasses = answer.titles
            expressionScores = answer.probabilities
            showFace = answer.labels[answer.top] != "no_face"
            let rect = showFace ? boxes.first { $0.name == "face_box" }?.rect ?? .zero : .zero
            face = FaceResult(rect: rect, probs: answer.probabilities, valence: 0, arousal: 0, topIndex: answer.top)
        }
        if let answer = choices.first(where: { $0.name == "hand" }) {
            gestureClasses = answer.labels
            gestureScores = answer.probabilities
            gestureRows = ScoreMath.stickyTopRows(gestureScores, previous: gestureRows, count: gestureRowCount, margin: 0.06)
            let showHand = answer.labels[answer.top] != "no_gesture"
            let rect = showHand ? boxes.first { $0.name == "hand_box" }?.rect ?? .zero : .zero
            let hand = HandResult(rect: rect, scores: answer.probabilities, classIndex: answer.top,
                                  score: answer.probabilities[answer.top])
            hands = [hand]
            handTiles = showHand ? result.handCrop.map { [HandTile(hand: hand, image: $0)] } ?? [] : []
        }
        faceCrop = showFace ? result.faceCrop : nil

        let a = 0.3
        func smooth(_ old: Double, _ new: Double) -> Double { old == 0 ? new : old + (new - old) * a }
        d1VisionMs = smooth(d1VisionMs, result.visionMs)
        d1BackboneMs = smooth(d1BackboneMs, result.backboneMs)
        d1TotalMs = smooth(d1TotalMs, result.totalMs)
        d1Tokens = result.tokens
        if let latency = result.latencyMs { latencyMs = smooth(latencyMs, latency) }
        if let delay = result.cameraDelayMs { cameraDelayMs = smooth(cameraDelayMs, delay) }
        let now = CACurrentMediaTime()
        if d1LastResult > 0 { inferenceFPS = smooth(inferenceFPS, 1 / max(now - d1LastResult, 1e-3)) }
        d1LastResult = now
        logStatsIfDue()
    }

    func start() {
        guard !started else { return }
        started = true

        cameras = CameraController.availableDevices().map { CameraOption(id: $0.uniqueID, name: $0.localizedName) }
        selectedCameraID = CameraController.preferredDevice()?.uniqueID ?? cameras.first?.id ?? ""

        // Optional launch overrides for scripted A/B runs, e.g.
        // open "ANE Classifiers.app" --args -computeMode GPU -faceDetector Vision -uncapped YES
        let defaults = UserDefaults.standard
        if let raw = defaults.string(forKey: "computeMode"), let mode = ComputeMode(rawValue: raw) { computeMode = mode }
        if let raw = defaults.string(forKey: "faceDetector"), let kind = FaceDetectorKind(rawValue: raw) {
            faceDetector = kind
            engine.setFaceDetector(kind)
        }
        if defaults.bool(forKey: "uncapped") {
            uncapped = true
            engine.setUncapped(true)
        }
        if defaults.bool(forKey: "d1") { setPipeline(.d1) }
        if let raw = defaults.string(forKey: "aneWake"), let mode = ANEWakeMode(rawValue: raw) { aneWake = mode }
        engine.setWakeMode(aneWake)
        let lead = defaults.double(forKey: "aneWakeLeadMs")
        if lead > 0 { engine.wakeLead = lead / 1000 }
        loadModels()

        CameraController.requestAccess { [weak self] granted in
            DispatchQueue.main.async {
                guard let self else { return }
                if granted {
                    self.startCamera()
                } else {
                    self.status = "Camera access denied. Allow it in System Settings › Privacy & Security › Camera."
                }
            }
        }
    }

    func setComputeMode(_ mode: ComputeMode) {
        guard mode != computeMode else { return }
        computeMode = mode
        loadModels()
    }

    func setFaceDetector(_ kind: FaceDetectorKind) {
        guard kind != faceDetector else { return }
        faceDetector = kind
        timings = StageTimings()  // restart the smoothed timings so the new detector's cost shows at once
        engine.setFaceDetector(kind)
    }

    func setUncapped(_ on: Bool) {
        guard on != uncapped else { return }
        uncapped = on
        timings = StageTimings()
        engine.setUncapped(on)
    }

    func setANEWake(_ mode: ANEWakeMode) {
        guard mode != aneWake else { return }
        aneWake = mode
        timings = StageTimings()
        engine.setWakeMode(mode)
    }

    func setHighFrameRate(_ on: Bool) {
        guard on != highFrameRate else { return }
        highFrameRate = on
        startCamera()
    }

    func selectCamera(_ id: String) {
        guard id != selectedCameraID else { return }
        selectedCameraID = id
        startCamera()
    }

    // MARK: - Private

    private func loadModels() {
        let mode = computeMode
        status = "Loading models for \(mode.rawValue)…"
        engine.load(mode: mode) { [weak self] result in
            DispatchQueue.main.async {
                guard let self, self.computeMode == mode else { return }
                switch result {
                case .success(let info):
                    self.expressionClasses = info.expressionClasses
                    self.gestureClasses = info.gestureClasses
                    self.loadSeconds = info.seconds
                    self.timings = StageTimings()
                    self.status = "Running on \(mode.rawValue)"
                    self.logger.info("Loaded models for \(mode.rawValue, privacy: .public) in \(info.seconds, format: .fixed(precision: 2), privacy: .public) s")
                case .failure(let error):
                    self.status = "Model load failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func startCamera() {
        camera.start(deviceID: selectedCameraID, highFrameRate: highFrameRate) { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                switch result {
                case .success(let info):
                    self.cameraMaxFPS = info.maxFrameRate
                    self.logger.info("Camera running at \(info.activeFrameRate, format: .fixed(precision: 0), privacy: .public) fps (max \(info.maxFrameRate, format: .fixed(precision: 0), privacy: .public)); system effects: \(info.systemEffects, privacy: .public)")
                case .failure(let error):
                    self.status = "Camera error: \(error.localizedDescription)"
                }
            }
        }
    }

    /// Camera video queue.
    private func handleFrame(_ sampleBuffer: CMSampleBuffer) {
        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: true),
           CFArrayGetCount(attachments) > 0 {
            let dict = unsafeBitCast(CFArrayGetValueAtIndex(attachments, 0), to: CFMutableDictionary.self)
            CFDictionarySetValue(dict,
                                 Unmanaged.passUnretained(kCMSampleAttachmentKey_DisplayImmediately).toOpaque(),
                                 Unmanaged.passUnretained(kCFBooleanTrue).toOpaque())
        }
        if renderer.status == .failed { renderer.flush() }
        renderer.enqueue(sampleBuffer)

        cameraFrameCount += 1
        let now = CACurrentMediaTime()
        if now - cameraWindowStart >= 0.5 {
            let rate = Double(cameraFrameCount) / (now - cameraWindowStart)
            cameraFrameCount = 0
            cameraWindowStart = now
            DispatchQueue.main.async { [weak self] in self?.cameraFPS = rate }
        }

        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            // Capture time on the host clock (same base as CACurrentMediaTime) for latency stats.
            var captureTime: CFTimeInterval = 0
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            if pts.isValid {
                let host = camera.session.synchronizationClock
                    .map { CMSyncConvertTime(pts, from: $0, to: CMClockGetHostTimeClock()) } ?? pts
                captureTime = host.seconds
            }
            engine.submit(pixelBuffer, captureTime: captureTime)  // ignored while paused (d1 selected)
            d1.submit(pixelBuffer, captureTime: captureTime)      // ignored while d1 is off
        }
    }

    private func apply(_ result: FrameResult) {
        guard pipeline == .classic else { return }
        frameSize = result.frameSize
        face = result.face
        hands = result.hands
        faceCrop = result.faceCrop
        handTiles = result.handTiles

        // Scores arrive already smoothed per face / hand track (Stabilizer); only fade out when gone.
        if let face = result.face {
            expressionScores = face.probs
            valence = face.valence
            arousal = face.arousal
        } else {
            expressionScores = expressionScores.map { $0 * fade }
            valence *= fade
            arousal *= fade
        }
        gestureScores = result.hands.first?.scores ?? gestureScores.map { $0 * fade }
        gestureRows = ScoreMath.stickyTopRows(gestureScores, previous: gestureRows, count: gestureRowCount, margin: 0.06)

        let t = result.timings, a = timingSmoothing
        if timings.total == 0 {
            timings = t
        } else {
            timings.letterbox += (t.letterbox - timings.letterbox) * a
            timings.faceDetect += (t.faceDetect - timings.faceDetect) * a
            timings.expression += (t.expression - timings.expression) * a
            timings.gesture += (t.gesture - timings.gesture) * a
            timings.total += (t.total - timings.total) * a
        }
        if let fps = result.inferenceFPS { inferenceFPS = fps }
        if let latency = result.latencyMs {
            latencyMs = latencyMs == 0 ? latency : latencyMs + (latency - latencyMs) * timingSmoothing
        }
        if let delay = result.cameraDelayMs {
            cameraDelayMs = cameraDelayMs == 0 ? delay : cameraDelayMs + (delay - cameraDelayMs) * timingSmoothing
        }
        logStatsIfDue()
    }

    /// One compact line every few seconds (Console.app, subsystem com.example.ANEDemo).
    private func logStatsIfDue() {
        let now = CACurrentMediaTime()
        guard now - lastStatsLog >= 3 else { return }
        lastStatsLog = now
        let faceText = face.map { f in
            let name = expressionClasses.indices.contains(f.topIndex) ? expressionClasses[f.topIndex] : "?"
            return String(format: "%@ %.2f v%+.2f a%+.2f", name, f.probs[f.topIndex], f.valence, f.arousal)
        } ?? "none"
        let handText = hands.isEmpty ? "none" : hands.map { h in
            let name = gestureClasses.indices.contains(h.classIndex) ? gestureClasses[h.classIndex] : "?"
            return String(format: "%@ %.2f", name, h.score)
        }.joined(separator: ", ")
        let line: String
        if pipeline == .d1 {
            line = String(format: "[d1-2B · GPU] vision %.0f | backbone %.0f | total %.0f ms | %d tokens | %.1f dec/s | capture→decision %.0f ms",
                          d1VisionMs, d1BackboneMs, d1TotalMs, d1Tokens, inferenceFPS, latencyMs)
        } else {
            let t = timings
            line = String(format: "[%@ · %@] face %.1f | expr %.1f | letterbox %.1f | gesture %.1f | total %.1f ms | %.0f dec/s | cam %.0f fps | %dx%d",
                          computeMode.rawValue, faceDetector.rawValue, t.faceDetect, t.expression, t.letterbox, t.gesture, t.total,
                          inferenceFPS, cameraFPS, Int(frameSize.width), Int(frameSize.height))
        }
        logger.info("\(line, privacy: .public) | face: \(faceText, privacy: .public) | hands: \(handText, privacy: .public)")
    }
}
