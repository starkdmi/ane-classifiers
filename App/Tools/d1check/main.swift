// Numeric check of the Swift d1 port against mlx-vlm reference tensors.
// Usage: d1check MODEL_DIR PLAN_JSON REFERENCE_SAFETENSORS
import Foundation
import MLX

func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

let args = CommandLine.arguments
guard args.count == 4 else {
    print("usage: d1check MODEL_DIR PLAN_JSON REFERENCE_SAFETENSORS")
    exit(2)
}
let plan = try JSONDecoder().decode(D1Plan.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
let reference = try loadArrays(url: URL(fileURLWithPath: args[3]))

var start = Date()
let runner = try D1Runner(modelDirectory: URL(fileURLWithPath: args[1]), plan: plan)
print(String(format: "loaded in %.1f s · %d tokens · grid %dx%d", Date().timeIntervalSince(start),
             runner.tokenCount, runner.gridHeight, runner.gridWidth))

// 1. preprocessing: same RGB in, same patches out
let pixels = runner.preprocess(rgb: reference["image_rgb"]!)
print("pixel_values  max|Δ| =", maxAbs(pixels, reference["pixel_values"]!), pixels.shape)

// 2. vision tower + projector on the reference patches
let features = runner.model.encodeImage(reference["pixel_values"]!, gridHeight: runner.gridHeight, gridWidth: runner.gridWidth)
print("features      max|Δ| =", maxAbs(features, reference["features"]!), features.shape)

// 3. backbone under set attention: hidden rows at the answer slots
let slots = runner.slotHidden(pixelValues: reference["pixel_values"]!)
let refSlots = reference["hidden_slots"]!
print("hidden_slots  max|Δ| =", maxAbs(slots, refSlots),
      " (ref scale", abs(refSlots).max().item(Float.self), ")")

// 4. final answers
let decision = runner.decide(pixelValues: pixels)
for answer in decision.choices {
    let ref = reference["probs_\(answer.name)"]!.asArray(Float.self)
    let delta = zip(ref, answer.probabilities).map { abs($0 - $1) }.max() ?? 0
    let refTop = ref.indices.max { ref[$0] < ref[$1] }!
    print(String(format: "%@: swift %@ %.3f | python %@ %.3f | max|Δp| %.4f", answer.name,
                 answer.labels[answer.top], answer.probabilities[answer.top], answer.labels[refTop], ref[refTop], delta))
}
for box in decision.boxes {
    let ref = reference["box_\(box.name)"]!.asArray(Float.self)  // found, top, bottom, left, right
    let r = box.rect ?? .zero
    let swift = [Float(box.found), Float(r.minY), Float(r.maxY), Float(r.minX), Float(r.maxX)]
    let delta = zip(ref, swift).map { abs($0 - $1) }.max() ?? 0
    print(String(format: "%@: swift found %.3f t/b/l/r %.3f %.3f %.3f %.3f | python found %.3f %.3f %.3f %.3f %.3f | max|Δ| %.4f",
                 box.name, swift[0], swift[1], swift[2], swift[3], swift[4], ref[0], ref[1], ref[2], ref[3], ref[4], delta))
}

// 5. speed (warm, preprocessing included)
var times: [Double] = []
for _ in 0 ..< 6 {
    start = Date()
    _ = runner.decide(pixelValues: runner.preprocess(rgb: reference["image_rgb"]!))
    times.append(Date().timeIntervalSince(start) * 1000)
}
times = Array(times.dropFirst()).sorted()
print(String(format: "decide (Swift MLX): median %.0f ms, min %.0f ms", times[times.count / 2], times[0]))
