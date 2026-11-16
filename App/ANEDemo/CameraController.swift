import AVFoundation
import CoreGraphics

enum CameraError: LocalizedError {
    case noCamera
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .noCamera: return "No camera found"
        case .cannotAddInput: return "Camera input could not be added"
        case .cannotAddOutput: return "Video output could not be added"
        }
    }
}

/// Owns the capture session. Frames arrive mirrored (selfie view) as 32BGRA on `videoQueue`.
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let videoQueue = DispatchQueue(label: "camera.video", qos: .userInteractive)
    /// Called on `videoQueue` for every frame.
    var frameHandler: ((CMSampleBuffer) -> Void)?

    private let output = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private var currentInput: AVCaptureDeviceInput?

    static func availableDevices() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
            mediaType: .video,
            position: .unspecified
        ).devices
    }

    static func preferredDevice() -> AVCaptureDevice? {
        let devices = availableDevices()
        return devices.first { $0.deviceType == .builtInWideAngleCamera }
            ?? AVCaptureDevice.default(for: .video)
            ?? devices.first
    }

    static func requestAccess(_ completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: completion(true)
        case .notDetermined: AVCaptureDevice.requestAccess(for: .video, completionHandler: completion)
        default: completion(false)
        }
    }

    struct Info {
        /// Fastest frame rate any (≤1080p, landscape) format of this camera offers.
        var maxFrameRate: Double
        /// Frame rate the session is running at now.
        var activeFrameRate: Double
        /// System camera effects as macOS reports them for this app (user-controlled in the Video menu).
        var systemEffects: String
    }

    /// The on/off state of macOS's own camera effects for this app. Apps can read but not switch these
    /// (Reactions gestures only get a default, via NSCameraReactionEffectGesturesEnabledDefault).
    static func systemEffectsSummary() -> String {
        var parts = [
            "portrait \(AVCaptureDevice.isPortraitEffectEnabled)",
            "centerStage \(AVCaptureDevice.isCenterStageEnabled)",
            "studioLight \(AVCaptureDevice.isStudioLightEnabled)",
            "reactions \(AVCaptureDevice.reactionEffectsEnabled)",
            "reactionGestures \(AVCaptureDevice.reactionEffectGesturesEnabled)",
        ]
        if #available(macOS 15.0, *) {
            parts.append("backgroundReplacement \(AVCaptureDevice.isBackgroundReplacementEnabled)")
        }
        return parts.joined(separator: ", ")
    }

    /// (Re)configures the session for `deviceID` and starts it. With `highFrameRate`, the camera's
    /// fastest format is used when it beats 30 fps. `completion` runs on a background queue.
    func start(deviceID: String?, highFrameRate: Bool, completion: @escaping (Result<Info, Error>) -> Void) {
        sessionQueue.async { [self] in
            do {
                let info = try configure(deviceID: deviceID, highFrameRate: highFrameRate)
                if !session.isRunning { session.startRunning() }
                completion(.success(info))
            } catch {
                completion(.failure(error))
            }
        }
    }

    /// The landscape format (≤1920 wide) with the highest frame rate; larger frames win ties.
    private static func fastestFormat(_ device: AVCaptureDevice) -> (format: AVCaptureDevice.Format, range: AVFrameRateRange)? {
        device.formats.compactMap { format -> (AVCaptureDevice.Format, AVFrameRateRange, Int32)? in
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard d.width <= 1920, d.width >= d.height,
                  let range = format.videoSupportedFrameRateRanges.max(by: { $0.maxFrameRate < $1.maxFrameRate }) else { return nil }
            return (format, range, d.width)
        }
        .max { ($0.1.maxFrameRate, $0.2) < ($1.1.maxFrameRate, $1.2) }
        .map { ($0.0, $0.1) }
    }

    private func configure(deviceID: String?, highFrameRate: Bool) throws -> Info {
        let device = Self.availableDevices().first { $0.uniqueID == deviceID } ?? Self.preferredDevice()
        guard let device else { throw CameraError.noCamera }

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        if let currentInput { session.removeInput(currentInput) }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.cannotAddInput }
        session.addInput(input)
        currentInput = input

        // Native resolution keeps the preview sharp; the pipeline downsamples on its own (one vImage pass).
        let fastest = Self.fastestFormat(device)
        if highFrameRate, let fastest, fastest.range.maxFrameRate > 30.5 {
            try device.lockForConfiguration()
            device.activeFormat = fastest.format  // switches the session to .inputPriority
            device.activeVideoMinFrameDuration = fastest.range.minFrameDuration
            device.activeVideoMaxFrameDuration = fastest.range.minFrameDuration
            device.unlockForConfiguration()
        } else {
            session.sessionPreset = .high
        }

        if !session.outputs.contains(output) {
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: videoQueue)
            guard session.canAddOutput(output) else { throw CameraError.cannotAddOutput }
            session.addOutput(output)
        }

        // Mirror the pixels themselves, so display, inference and boxes all share one coordinate space.
        if let connection = output.connection(with: .video), connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = true
        }

        let minDuration = device.activeVideoMinFrameDuration
        let active = minDuration.isValid && minDuration.seconds > 0 ? 1 / minDuration.seconds : 30
        return Info(maxFrameRate: fastest?.range.maxFrameRate ?? active, activeFrameRate: active,
                    systemEffects: Self.systemEffectsSummary())
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameHandler?(sampleBuffer)
    }
}
