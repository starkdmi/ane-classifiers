// The set-attention masks, the answer readout, the calibration ("through") and the box reading (edge medians,
// crossed-edge ordering) are adapted from mlx-vlm's models/d1_lfm3_vl (itself adapted from the system_one code
// shipped with the LiquidAI d1 checkpoints): LFM Open License v1.0, see LICENSE in this folder. This file is
// distributed under that license, not under the repository's MIT license.
//
// d1 System One for a fixed, pre-compiled question set ("plan", exported by tools/d1_export_plan.py).
// Per frame only the image changes: preprocess -> vision tower + projector -> one packed row through
// the LFM3 backbone under set attention -> logits at each answer slot over its readout table ->
// choice probabilities (calibrated softmax) and boxes (presence + four edge medians).

import CoreGraphics
import Foundation
import MLX
import MLXNN

struct D1Plan: Decodable {
    struct Image: Decodable {
        var width: Int
        var height: Int
        var grid_thw: [Int]
        var mean: [Float]
        var std: [Float]
        var patch: Int
        var temporal: Int
        var merge: Int
    }

    struct Table: Decodable {
        var tokens: [Int]
        var forms: [[Int]]
    }

    struct Slot: Decodable {
        var at: Int
        var table: Int
        var temperature: Float
    }

    struct Grid: Decodable {
        var low: Double
        var high: Double
        var size: Int

        var step: Double { (high - low) / Double(size - 1) }
        var values: [Double] { (0 ..< size - 1).map { low + Double($0) * step } + [high] }
    }

    struct Question: Decodable {
        var name: String
        var kind: String  // "choice" or "box"
        var slots: [Slot]
        var labels: [String]?
        var titles: [String]?
        var found_knots: [[Double]]?
        var grid: Grid?
        var edges: String?
    }

    var model: String
    var image: Image
    var image_token_id: Int
    var sliding_window: Int
    var ids: [Int]
    var state_len: Int
    var question: [Int]
    var tables: [Table]
    var questions: [Question]
}

struct D1Answer {
    var name: String
    var labels: [String]
    var titles: [String]
    var probabilities: [Float]

    var top: Int { probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0 }
}

struct D1Box {
    var name: String
    /// Calibrated presence probability.
    var found: Double
    /// Normalized, top-left origin, in the picture d1 saw (the whole frame); nil when not found.
    var rect: CGRect?
}

struct D1Decision {
    var choices: [D1Answer]
    var boxes: [D1Box]
}

/// Not thread-safe: drive it from one queue.
final class D1Runner {
    let model: D1Model
    let plan: D1Plan
    var gridHeight: Int { plan.image.grid_thw[1] }
    var gridWidth: Int { plan.image.grid_thw[2] }
    var tokenCount: Int { plan.ids.count }

    private let ids: MLXArray
    private let imagePositions: MLXArray
    private let masks: [String: MLXArray?]
    private let ropes: [String: (MLXArray, MLXArray)]
    /// Every answer slot, in plan order (question by question).
    private let slotPositions: MLXArray
    private let slotList: [D1Plan.Slot]
    private let tableEmbeddings: [MLXArray]
    /// For each table: which rows of slotList read it.
    private let tableRows: [(table: Int, rows: [Int], index: MLXArray)]

    init(modelDirectory: URL, plan: D1Plan) throws {
        let model = try D1Model.load(directory: modelDirectory)
        self.model = model
        self.plan = plan

        let length = plan.ids.count
        ids = MLXArray(plan.ids.map { Int32($0) })
        imagePositions = MLXArray(plan.ids.indices.filter { plan.ids[$0] == plan.image_token_id }.map { Int32($0) })

        // Set attention: full layers let the state and each question see each other (both ways);
        // sliding layers stay causal within their window; convolutions are causal by construction.
        let positions = MLXArray((0 ..< length).map { Int32($0) })
        let rows = positions.expandedDimensions(axis: 1), cols = positions.expandedDimensions(axis: 0)
        let group = MLXArray(plan.question.map { Int32($0) })
        let state = MLXArray(Int32(plan.state_len))
        let full = logicalOr(
            logicalOr(rows .< state, cols .< state),
            group.expandedDimensions(axis: 1) .== group.expandedDimensions(axis: 0))
        var sliding = rows .>= cols
        if length > plan.sliding_window {
            sliding = logicalAnd(sliding, rows .< cols + Int32(plan.sliding_window))
        }
        masks = ["full_attention": full, "sliding_attention": sliding, "conv": nil]

        var ropes: [String: (MLXArray, MLXArray)] = [:]
        for layer in model.languageModel.model.layers {
            if let attention = layer.selfAttn, ropes[layer.layerType] == nil {
                ropes[layer.layerType] = attention.rotary(length: length)
            }
        }
        self.ropes = ropes

        let slots = plan.questions.flatMap(\.slots)
        slotList = slots
        slotPositions = MLXArray(slots.map { Int32($0.at) })
        let embed = model.languageModel.model.embedTokens
        tableEmbeddings = plan.tables.map { embed(MLXArray($0.tokens.map { Int32($0) })) }
        tableRows = plan.tables.indices.compactMap { t in
            let rows = slots.indices.filter { slots[$0].table == t }
            return rows.isEmpty ? nil : (t, rows, MLXArray(rows.map { Int32($0) }))
        }
        eval([ids, imagePositions, slotPositions, full, sliding] + tableEmbeddings + ropes.values.flatMap { [$0.0, $0.1] })
    }

    /// RGB image [H, W, 3] (UInt8, already plan.image.width x height) -> patches [gridH*gridW, 3*T*p*p],
    /// in the LFM3-VL / Qwen-VL processor's order (each picture repeated over the temporal patch).
    func preprocess(rgb: MLXArray) -> MLXArray {
        let image = plan.image
        let mean = MLXArray(image.mean), std = MLXArray(image.std)
        let x = ((rgb.asType(.float32) / 255) - mean) / std               // [H, W, 3]
        let chw = x.transposed(2, 0, 1)                                     // [3, H, W]
        let frames = stacked(Array(repeating: chw, count: image.temporal), axis: 0)  // [T, 3, H, W]
        let (p, m, gh, gw) = (image.patch, image.merge, gridHeight, gridWidth)
        return frames
            .reshaped(1, image.temporal, 3, gh / m, m, p, gw / m, m, p)
            .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshaped(gh * gw, 3 * image.temporal * p * p)
    }

    /// Hidden rows at every answer slot [slots, D], in plan order.
    func slotHidden(pixelValues: MLXArray) -> MLXArray {
        let features = model.encodeImage(pixelValues, gridHeight: gridHeight, gridWidth: gridWidth)
        let hidden = model.hidden(ids: ids, imagePositions: imagePositions, features: features, masks: masks, ropes: ropes)
        return hidden[slotPositions]
    }

    /// Time of the last decide(): vision tower + projector, then backbone + readout (ms).
    private(set) var lastVisionMs = 0.0
    private(set) var lastBackboneMs = 0.0

    /// One decision: every question (choices and boxes) answered from a single forward pass.
    func decide(pixelValues: MLXArray) -> D1Decision {
        let t0 = Date()
        let features = model.encodeImage(pixelValues, gridHeight: gridHeight, gridWidth: gridWidth)
        eval(features)
        let t1 = Date()
        let hidden = model.hidden(ids: ids, imagePositions: imagePositions, features: features, masks: masks, ropes: ropes)
        let hiddenRows = hidden[slotPositions]
        // One matmul per readout table over the slots that read it (a box's four edges share one).
        let logits = tableRows.map { matmul(hiddenRows[$0.index], tableEmbeddings[$0.table].T).asType(.float32) }
        eval(logits)
        lastVisionMs = t1.timeIntervalSince(t0) * 1000
        lastBackboneMs = Date().timeIntervalSince(t1) * 1000

        var distributions = [[Double]](repeating: [], count: slotList.count)
        for (entry, tableLogits) in zip(tableRows, logits) {
            let forms = plan.tables[entry.table].forms
            let width = tableLogits.dim(1)
            let values = tableLogits.asArray(Float.self)
            for (r, row) in entry.rows.enumerated() {
                let temperature = Double(slotList[row].temperature)
                // Each option scores its best token form, then the temperature, then a softmax.
                let scores = forms.map { group in Double(group.map { values[r * width + $0] }.max()!) / temperature }
                let peak = scores.max()!
                let exps = scores.map { Foundation.exp($0 - peak) }
                let total = exps.reduce(0, +)
                distributions[row] = exps.map { $0 / total }
            }
        }

        var decision = D1Decision(choices: [], boxes: [])
        var row = 0
        for question in plan.questions {
            let own = Array(distributions[row ..< row + question.slots.count])
            row += question.slots.count
            if question.kind == "box" {
                decision.boxes.append(Self.box(question, own))
            } else {
                decision.choices.append(D1Answer(name: question.name, labels: question.labels ?? [],
                                                 titles: question.titles ?? [], probabilities: own[0].map { Float($0) }))
            }
        }
        return decision
    }

    // MARK: - Box answer (mirrors mlx-vlm d1 box.answer with edges read at the median)

    private static func box(_ q: D1Plan.Question, _ dists: [[Double]]) -> D1Box {
        let found = through(dists[0][0], knots: q.found_knots ?? [])
        guard found >= 0.5, let grid = q.grid, dists.count == 5 else { return D1Box(name: q.name, found: found, rect: nil) }
        let values = grid.values
        let edges = dists[1...].map { median($0, values: values) }
        let (top, bottom) = ordered(edges[0], edges[1], step: grid.step)
        let (left, right) = ordered(edges[2], edges[3], step: grid.step)
        return D1Box(name: q.name, found: found, rect: CGRect(x: left, y: top, width: right - left, height: bottom - top))
    }

    /// The first grid value whose cumulative probability reaches one half.
    private static func median(_ probabilities: [Double], values: [Double]) -> Double {
        let total = probabilities.reduce(0, +)
        var mass = 0.0
        for (value, p) in zip(values, probabilities) {
            mass += p / total
            if mass >= 0.5 { return value }
        }
        return values.last ?? 0
    }

    /// Crossed edges become one bin about their middle.
    private static func ordered(_ a: Double, _ b: Double, step: Double) -> (Double, Double) {
        if a < b { return (a, b) }
        let half = step / 2
        let middle = min(1 - half, max(half, (a + b) / 2))
        return (middle - half, middle + half)
    }

    /// A value through increasing (raw, calibrated) knots, linear between them, flat past the ends.
    private static func through(_ value: Double, knots: [[Double]]) -> Double {
        guard let first = knots.first, let last = knots.last else { return value }
        guard let index = knots.firstIndex(where: { $0[0] >= value }) else { return last[1] }
        if index == 0 { return first[1] }
        let (a, b) = (knots[index - 1], knots[index])
        return a[1] + (value - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
    }
}
