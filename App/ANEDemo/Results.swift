import CoreGraphics
import CoreML

/// Which silicon the Core ML models (and Vision's face detector) are asked to run on.
enum ComputeMode: String, CaseIterable, Identifiable {
    case neuralEngine = "ANE"
    case hybrid = "Hybrid"
    case gpu = "GPU"
    case cpu = "CPU"
    case auto = "Auto"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .neuralEngine: return "Neural Engine"
        case .hybrid: return "Hybrid: gestures on GPU ∥ face + expression on ANE"
        case .gpu: return "GPU"
        case .cpu: return "CPU"
        case .auto: return "Auto (Core ML decides)"
        }
    }

    /// Face detector + expression model.
    var units: MLComputeUnits {
        switch self {
        case .neuralEngine, .hybrid: return .cpuAndNeuralEngine
        case .gpu: return .cpuAndGPU
        case .cpu: return .cpuOnly
        case .auto: return .all
        }
    }

    /// Gesture model: on the GPU in Hybrid, so it can run at the same time as the ANE models.
    var gestureUnits: MLComputeUnits { self == .hybrid ? .cpuAndGPU : units }
}

/// Which pipeline drives the app (only one runs at a time).
enum PipelineKind: String, CaseIterable, Identifiable {
    case classic = "Classic"
    case d1 = "d1-2B"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .classic: return "Classic: Core ML classifiers (BlazeFace · EmotiEffLib · YOLOv10n)"
        case .d1: return "d1-2B decision model (MLX on GPU, 2.2 GB)"
        }
    }
}

/// Hides the Neural Engine's wake-up cost (it idles between camera frames) with tiny early requests.
enum ANEWakeMode: String, CaseIterable, Identifiable {
    case off
    case arrival      // ping as a frame arrives, overlapping the CPU downscale
    case predictive   // ping shortly before the next frame is expected

    var id: String { rawValue }

    var title: String {
        switch self {
        case .off: return "Off"
        case .arrival: return "At frame arrival (overlaps downscale)"
        case .predictive: return "Before next frame (predicted)"
        }
    }

    var shortName: String {
        switch self {
        case .off: return "wake off"
        case .arrival: return "wake at arrival"
        case .predictive: return "wake predicted"
        }
    }
}

enum FaceDetectorKind: String, CaseIterable, Identifiable {
    case blazeFace = "BlazeFace"
    case vision = "Vision"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .blazeFace: return "BlazeFace (MediaPipe, Core ML)"
        case .vision: return "Vision (Apple built-in)"
        }
    }
}

/// All rects are normalized (0...1) with a top-left origin, in the (mirrored) camera frame.
struct FaceResult {
    var rect: CGRect
    var probs: [Float]
    var valence: Float
    var arousal: Float
    /// Displayed class; sticky (hysteresis) once the Stabilizer has seen a few frames.
    var topIndex: Int
}

struct HandResult {
    var rect: CGRect
    /// Independent sigmoid score per gesture class (YOLO style, not a softmax).
    var scores: [Float]
    var classIndex: Int
    var score: Float
}

struct StageTimings {
    var letterbox = 0.0
    var faceDetect = 0.0
    var expression = 0.0
    var gesture = 0.0
    var total = 0.0
}

/// A zoomed hand for the tile row: up to two hands, ordered left-to-right on screen.
struct HandTile {
    var hand: HandResult
    var image: CGImage?
}

struct FrameResult {
    var frameSize: CGSize
    var face: FaceResult?
    var hands: [HandResult]
    var faceCrop: CGImage?
    var handTiles: [HandTile]
    var timings: StageTimings
    var inferenceFPS: Double?
    /// Camera capture -> result ready; nil when this run re-used an already processed frame.
    var latencyMs: Double?
    /// Camera capture -> frame delivered to the app (sensor, ISP, driver); nil for re-used frames.
    var cameraDelayMs: Double?
}

struct LoadInfo {
    var seconds: Double
    var expressionClasses: [String]
    var gestureClasses: [String]
}

struct CameraOption: Identifiable, Hashable {
    var id: String
    var name: String
}

enum Labels {
    static func expression(_ name: String) -> (emoji: String, title: String) {
        let emoji: [String: String] = [
            "Anger": "😠", "Contempt": "😏", "Disgust": "🤢", "Fear": "😨",
            "Happiness": "😄", "Neutral": "😐", "Sadness": "😢", "Surprise": "😮",
            // d1's options (shown by their titles)
            "Smiling": "😄", "Sad": "😢", "Surprised": "😮", "Thinking": "🤔", "No face visible": "·",
        ]
        return (emoji[name] ?? "🙂", name)
    }

    static func gesture(_ key: String) -> (emoji: String, title: String) {
        let known: [String: (String, String)] = [
            "like": ("👍", "Thumbs up"),
            "dislike": ("👎", "Thumbs down"),
            "ok": ("👌", "OK"),
            "peace": ("✌️", "Peace"),
            "peace_inverted": ("✌️", "Peace, inverted"),
            "two_up": ("✌️", "Two up"),
            "two_up_inverted": ("✌️", "Two up, inverted"),
            "palm": ("✋", "Palm"),
            "stop": ("🤚", "Stop"),
            "stop_inverted": ("🤚", "Stop, inverted"),
            "fist": ("✊", "Fist"),
            "grip": ("✊", "Grip"),
            "one": ("☝️", "One"),
            "point": ("👉", "Point"),
            "three": ("🤟", "Three"),
            "three2": ("🤟", "Three (alt)"),
            "three3": ("🤟", "Three (alt 2)"),
            "three_gun": ("🔫", "Three gun"),
            "four": ("🖖", "Four"),
            "call": ("🤙", "Call me"),
            "rock": ("🤘", "Rock"),
            "mute": ("🤫", "Mute"),
            "hand_heart": ("🫶", "Hand heart"),
            "hand_heart2": ("🫶", "Hand heart (alt)"),
            "holy": ("🙏", "Holy"),
            "middle_finger": ("🖕", "Middle finger"),
            "little_finger": ("🤙", "Little finger"),
            "grabbing": ("🫳", "Grabbing"),
            "take_picture": ("📸", "Take picture"),
            "thumb_index": ("🤏", "Thumb + index"),
            "thumb_index2": ("🤏", "Thumb + index (alt)"),
            "timeout": ("⏸️", "Timeout"),
            "xsign": ("❌", "X sign"),
            "no_gesture": ("·", "No gesture"),
            // d1's options
            "thumbs_up": ("👍", "Thumbs up"),
            "thumbs_down": ("👎", "Thumbs down"),
            "open_palm": ("✋", "Open palm"),
            "pointing": ("☝️", "Pointing"),
            "heart": ("🫶", "Heart"),
        ]
        if let (emoji, title) = known[key] { return (emoji, title) }
        return ("✋", key.replacingOccurrences(of: "_", with: " ").capitalized)
    }
}
