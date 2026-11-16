// Swift/MLX port of LiquidAI d1-2B (LFM3-VL backbone): the forward pass System One needs (no cache).
//
// Ported from mlx-vlm (https://github.com/Blaizzy/mlx-vlm):
//   - models/lfm3, models/lfm3_vl: the LFM3 / LFM3-VL architecture. Copyright 2026 The HuggingFace Team,
//     adapted for MLX from the LFM3 reference implementations; Apache License 2.0 (LICENSES/Apache-2.0.txt).
//   - models/d1_lfm3_vl: d1's set-attention forward pass. LFM Open License v1.0 (LICENSE in this folder).
// This file is distributed under those terms, not under the repository's MIT license.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Config

struct D1Config: Decodable {
    struct Text: Decodable {
        var hidden_size: Int
        var intermediate_size: Int
        var num_hidden_layers: Int
        var num_attention_heads: Int
        var num_key_value_heads: Int
        var head_dim: Int
        var vocab_size: Int
        var norm_eps: Float
        var conv_L_cache: Int
        var conv_bias: Bool
        var layer_types: [String]
        var sliding_window: Int
        var use_sandwich_norm: Bool
        var rope_parameters: [String: RopeParameters]
    }

    struct RopeParameters: Decodable {
        var rope_theta: Double?
        var partial_rotary_factor: Double?
    }

    struct Vision: Decodable {
        var hidden_size: Int
        var intermediate_size: Int
        var num_hidden_layers: Int
        var num_attention_heads: Int
        var num_key_value_heads: Int
        var norm_eps: Float
        var patch_size: Int
        var temporal_patch_size: Int
        var spatial_merge_size: Int
        var in_channels: Int
        var conv_kernel_size: Int
        var layer_types: [String]
        var rope_parameters: RopeParameters
    }

    struct Quantization: Decodable {
        var group_size: Int
        var bits: Int
    }

    var text_config: Text
    var vision_config: Vision
    var downsample_factor: Int
    var projector_hidden_size: Int
    var projector_bias: Bool
    var projector_layer_norm_eps: Float
    var image_token_id: Int
    var quantization: Quantization?
}

// MARK: - Shared blocks

final class D1MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(hidden: Int, intermediate: Int) {
        _w1.wrappedValue = Linear(hidden, intermediate, bias: false)
        _w2.wrappedValue = Linear(intermediate, hidden, bias: false)
        _w3.wrappedValue = Linear(hidden, intermediate, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

/// Rotate matching coordinates in both head halves (LFM3's rotary layout).
@inline(__always)
private func rotateHalves(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let halves = split(x, parts: 2, axis: -1)
    return x * cos + concatenated([-halves[1], halves[0]], axis: -1) * sin
}

// MARK: - Vision tower

final class D1VisionAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

    init(_ c: D1Config.Vision) {
        heads = c.num_attention_heads
        kvHeads = c.num_key_value_heads
        headDim = c.hidden_size / heads
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(c.hidden_size, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _outProj.wrappedValue = Linear(heads * headDim, c.hidden_size, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.norm_eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.norm_eps)
    }

    /// x: [N, D]; cos/sin: [N, headDim] float32.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let n = x.dim(0)
        var q = qNorm(qProj(x).reshaped(n, heads, headDim))
        var k = kNorm(kProj(x).reshaped(n, kvHeads, headDim))
        let v = vProj(x).reshaped(n, kvHeads, headDim)
        let c = cos.expandedDimensions(axis: 1), s = sin.expandedDimensions(axis: 1)
        q = rotateHalves(q.asType(.float32), cos: c, sin: s).asType(x.dtype)
        k = rotateHalves(k.asType(.float32), cos: c, sin: s).asType(x.dtype)
        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: scale, mask: nil)
        return outProj(out[0].transposed(1, 0, 2).reshaped(n, -1))
    }
}

final class D1VisionConv: Module {
    let merge: Int
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "spatial_conv") var spatialConv: Conv2d
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ c: D1Config.Vision) {
        merge = c.spatial_merge_size
        _inProj.wrappedValue = Linear(c.hidden_size, 3 * c.hidden_size, bias: false)
        _spatialConv.wrappedValue = Conv2d(
            inputChannels: c.hidden_size, outputChannels: c.hidden_size,
            kernelSize: IntOrPair(c.conv_kernel_size), padding: IntOrPair(c.conv_kernel_size / 2),
            groups: c.hidden_size, bias: false)
        _outProj.wrappedValue = Linear(c.hidden_size, c.hidden_size, bias: false)
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let parts = split(inProj(x), parts: 3, axis: -1)
        let (b, c, v) = (parts[0], parts[1], parts[2])
        let m = merge
        var grid = (b * v)
            .reshaped(height / m, width / m, m, m, -1)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(1, height, width, -1)
        grid = spatialConv(grid)
        let out = grid
            .reshaped(height / m, m, width / m, m, -1)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(height * width, -1)
        return outProj(c * out)
    }
}

final class D1VisionLayer: Module {
    let isAttention: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: D1VisionAttention?
    @ModuleInfo(key: "conv") var conv: D1VisionConv?
    @ModuleInfo(key: "feed_forward") var feedForward: D1MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ c: D1Config.Vision, index: Int) {
        isAttention = c.layer_types[index] == "full_attention"
        _selfAttn.wrappedValue = isAttention ? D1VisionAttention(c) : nil
        _conv.wrappedValue = isAttention ? nil : D1VisionConv(c)
        _feedForward.wrappedValue = D1MLP(hidden: c.hidden_size, intermediate: c.intermediate_size)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.norm_eps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.norm_eps)
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = operatorNorm(x)
        let mixed = isAttention ? selfAttn!(h, cos: cos, sin: sin) : conv!(h, height: height, width: width)
        let x = x + mixed
        return x + feedForward(ffnNorm(x))
    }
}

final class D1VisionModel: Module {
    let config: D1Config.Vision
    @ModuleInfo(key: "patchifier") var patchifier: Linear
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [D1VisionLayer]

    init(_ c: D1Config.Vision) {
        config = c
        _patchifier.wrappedValue = Linear(
            c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size, c.hidden_size, bias: false)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.norm_eps)
        _layers.wrappedValue = (0 ..< c.num_hidden_layers).map { D1VisionLayer(c, index: $0) }
    }

    /// Axial 2D rotary angles for an HxW patch grid, in merge-grouped token order. Returns cos, sin [H*W, headDim].
    func rotary(height: Int, width: Int) -> (MLXArray, MLXArray) {
        let m = config.spatial_merge_size
        var coords: [Float] = []
        coords.reserveCapacity(height * width * 2)
        for gy in 0 ..< height / m {
            for gx in 0 ..< width / m {
                for my in 0 ..< m {
                    for mx in 0 ..< m {
                        coords.append(Float(gy * m + my))
                        coords.append(Float(gx * m + mx))
                    }
                }
            }
        }
        let spatial = config.hidden_size / config.num_attention_heads / 2
        let theta = Float(config.rope_parameters.rope_theta ?? 1e6)
        let exponents = MLXArray(stride(from: 0, to: spatial, by: 2).map { -Float($0) / Float(spatial) })
        let invFreq = pow(MLXArray(theta), exponents)  // [spatial/2]
        let grid = MLXArray(coords, [height * width, 2])
        let angles = (grid.expandedDimensions(axis: -1) * invFreq).reshaped(height * width, -1)
        let full = concatenated([angles, angles], axis: -1)
        return (cos(full), sin(full))
    }

    /// pixelValues: [H*W, C*T*p*p] patches for one image, grid H x W (in patches).
    func callAsFunction(_ pixelValues: MLXArray, height: Int, width: Int) -> MLXArray {
        let (c, s) = rotary(height: height, width: width)
        var x = embeddingNorm(patchifier(pixelValues.asType(embeddingNorm.weight.dtype)))
        for layer in layers {
            x = layer(x, height: height, width: width, cos: c, sin: s)
        }
        return x
    }
}

final class D1Projector: Module {
    let inputDim: Int
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ c: D1Config) {
        inputDim = c.vision_config.hidden_size * c.downsample_factor * c.downsample_factor
        _layerNorm.wrappedValue = LayerNorm(dimensions: c.vision_config.hidden_size, eps: c.projector_layer_norm_eps)
        _linear1.wrappedValue = Linear(inputDim, c.projector_hidden_size, bias: c.projector_bias)
        _linear2.wrappedValue = Linear(c.projector_hidden_size, c.text_config.hidden_size, bias: c.projector_bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(layerNorm(x).reshaped(-1, inputDim))))
    }
}

// MARK: - LFM3 language backbone

final class D1TextAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float
    let ropeTheta: Float, ropeAngles: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    init(_ c: D1Config.Text, layerType: String) {
        heads = c.num_attention_heads
        kvHeads = c.num_key_value_heads
        headDim = c.head_dim
        scale = pow(Float(headDim), -0.5)
        let rope = c.rope_parameters[layerType]
        ropeTheta = Float(rope?.rope_theta ?? 1e6)
        ropeAngles = Int(Double(headDim) * (rope?.partial_rotary_factor ?? 1.0)) / 2
        _qProj.wrappedValue = Linear(c.hidden_size, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, c.hidden_size, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.norm_eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.norm_eps)
        _gateProj.wrappedValue = Linear(c.hidden_size, heads, bias: false)
    }

    /// cos/sin [L, headDim] (float32) for positions 0..<L.
    func rotary(length: Int) -> (MLXArray, MLXArray) {
        var freqs = (0 ..< ropeAngles).map { pow(ropeTheta, -Float($0) * 2 / Float(headDim)) }
        freqs += Array(repeating: 0, count: headDim / 2 - ropeAngles)  // NoPE pairs stay as they are
        let positions = MLXArray((0 ..< length).map { Float($0) })
        let angles = positions.expandedDimensions(axis: -1) * MLXArray(freqs)
        let full = concatenated([angles, angles], axis: -1)
        return (cos(full), sin(full))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, rope: (MLXArray, MLXArray)) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let c = rope.0.asType(x.dtype), s = rope.1.asType(x.dtype)
        q = rotateHalves(q, cos: c, sin: s)
        k = rotateHalves(k, cos: c, sin: s)
        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        out = out.transposed(0, 2, 1, 3)
        out = out * sigmoid(gateProj(x)).expandedDimensions(axis: -1)
        return oProj(out.reshaped(b, l, -1))
    }
}

final class D1ShortConv: Module {
    let kernel: Int
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ c: D1Config.Text) {
        kernel = c.conv_L_cache
        _conv.wrappedValue = Conv1d(
            inputChannels: c.hidden_size, outputChannels: c.hidden_size, kernelSize: c.conv_L_cache,
            groups: c.hidden_size, bias: c.conv_bias)
        _inProj.wrappedValue = Linear(c.hidden_size, 3 * c.hidden_size, bias: c.conv_bias)
        _outProj.wrappedValue = Linear(c.hidden_size, c.hidden_size, bias: c.conv_bias)
    }

    /// Causal depthwise convolution over the row (no cache).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(inProj(x), parts: 3, axis: -1)
        let bx = padded(parts[0] * parts[2], widths: [0, IntOrPair((kernel - 1, 0)), 0])
        return outProj(parts[1] * conv(bx))
    }
}

final class D1DecoderLayer: Module {
    let layerType: String
    @ModuleInfo(key: "self_attn") var selfAttn: D1TextAttention?
    @ModuleInfo(key: "conv") var conv: D1ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: D1MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFfnNorm: RMSNorm?

    init(_ c: D1Config.Text, index: Int) {
        layerType = c.layer_types[index]
        let isAttention = layerType != "conv"
        _selfAttn.wrappedValue = isAttention ? D1TextAttention(c, layerType: layerType) : nil
        _conv.wrappedValue = isAttention ? nil : D1ShortConv(c)
        _feedForward.wrappedValue = D1MLP(hidden: c.hidden_size, intermediate: c.intermediate_size)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.norm_eps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.norm_eps)
        _postOperatorNorm.wrappedValue = c.use_sandwich_norm ? RMSNorm(dimensions: c.hidden_size, eps: c.norm_eps) : nil
        _postFfnNorm.wrappedValue = c.use_sandwich_norm ? RMSNorm(dimensions: c.hidden_size, eps: c.norm_eps) : nil
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, rope: (MLXArray, MLXArray)?) -> MLXArray {
        var h = operatorNorm(x)
        h = selfAttn.map { $0(h, mask: mask, rope: rope!) } ?? conv!(h)
        let x = x + (postOperatorNorm.map { $0(h) } ?? h)
        let f = feedForward(ffnNorm(x))
        return x + (postFfnNorm.map { $0(f) } ?? f)
    }
}

final class D1TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [D1DecoderLayer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ c: D1Config.Text) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocab_size, dimensions: c.hidden_size)
        _layers.wrappedValue = (0 ..< c.num_hidden_layers).map { D1DecoderLayer(c, index: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.norm_eps)
    }
}

final class D1LanguageModel: Module {
    @ModuleInfo(key: "model") var model: D1TextModel

    init(_ c: D1Config.Text) {
        _model.wrappedValue = D1TextModel(c)
    }
}

// MARK: - Whole model

final class D1Model: Module {
    let config: D1Config
    @ModuleInfo(key: "vision_tower") var visionTower: D1VisionModel
    @ModuleInfo(key: "multi_modal_projector") var projector: D1Projector
    @ModuleInfo(key: "language_model") var languageModel: D1LanguageModel

    init(_ c: D1Config) {
        config = c
        _visionTower.wrappedValue = D1VisionModel(c.vision_config)
        _projector.wrappedValue = D1Projector(c)
        _languageModel.wrappedValue = D1LanguageModel(c.text_config)
    }

    /// Loads config.json + *.safetensors (MLX format, optionally quantized) from a model folder.
    static func load(directory: URL) throws -> D1Model {
        let config = try JSONDecoder().decode(D1Config.self, from: Data(contentsOf: directory.appending(path: "config.json")))
        let model = D1Model(config)
        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        for file in files {
            for (key, value) in try loadArrays(url: file) { weights[key] = value }
        }
        if let q = config.quantization {
            // Swap Linear/Embedding for their quantized versions wherever the checkpoint stores scales.
            quantize(model: model, groupSize: q.group_size, bits: q.bits, mode: .affine,
                     filter: { path, _ in weights["\(path).scales"] != nil })
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    /// Image features [tokens, textHidden] for one image's patches.
    func encodeImage(_ pixelValues: MLXArray, gridHeight: Int, gridWidth: Int) -> MLXArray {
        projector(visionTower(pixelValues, height: gridHeight, width: gridWidth))
    }

    /// Final hidden states [T, D] of one packed row under d1's set attention.
    func hidden(ids: MLXArray, imagePositions: MLXArray, features: MLXArray,
                masks: [String: MLXArray?], ropes: [String: (MLXArray, MLXArray)]) -> MLXArray {
        let backbone = languageModel.model
        var h = backbone.embedTokens(ids)
        h[imagePositions] = features.asType(h.dtype)
        h = h.expandedDimensions(axis: 0)
        for layer in backbone.layers {
            let mask = masks[layer.layerType] ?? nil
            h = layer(h, mask: mask, rope: ropes[layer.layerType])
        }
        return backbone.embeddingNorm(h)[0]
    }
}
