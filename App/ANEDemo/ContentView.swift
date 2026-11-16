import SwiftUI

enum Theme {
    static let background = Color(red: 0.055, green: 0.06, blue: 0.07)
    static let card = Color.white.opacity(0.045)
    static let face = Color(red: 0.45, green: 0.72, blue: 1.0)
    static let hand = Color(red: 1.0, green: 0.72, blue: 0.3)
    static let mono = Font.system(size: 11, design: .monospaced)
}

struct ContentView: View {
    @Bindable var model: AppModel

    var body: some View {
        GeometryReader { geo in
            let wide = geo.size.width >= 1000
            VStack(alignment: .leading, spacing: 14) {
                header
                if wide {
                    HStack(alignment: .top, spacing: 16) {
                        // Video on top, zoom tiles in the space under it.
                        GeometryReader { column in
                            let layout = FeedLayout(available: column.size, frame: model.frameSize, gap: 12)
                            // Scrolls only when the window is too short for video + tiles.
                            ScrollView(.vertical) {
                                VStack(alignment: .leading, spacing: 12) {
                                    AnnotatedFeed(model: model)
                                        .frame(width: layout.feedSize.width, height: layout.feedSize.height)
                                    tileRow(side: layout.tileSide)
                                }
                                .frame(width: column.size.width, alignment: .leading)
                            }
                            .scrollIndicators(.automatic)
                        }
                        ScrollView { cards }
                            .scrollIndicators(.never)
                            .frame(width: 380)
                    }
                } else {
                    ScrollView {
                        VStack(spacing: 14) {
                            AnnotatedFeed(model: model)
                            tileRow(side: nil)
                            cards
                        }
                    }
                    .scrollIndicators(.never)
                }
                statsBar
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
        }
        .frame(minWidth: 640, minHeight: 560)
        .background(Theme.background)
        .preferredColorScheme(.dark)
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                let d1 = model.pipeline == .d1
                Text(d1 ? "WEBCAM → DECISIONS · MLX SWIFT · ZERO TOKENS GENERATED"
                        : "WEBCAM → CLASSIC CLASSIFIERS · CORE ML · ZERO TOKENS")
                    .font(Theme.mono)
                    .foregroundStyle(.secondary)
                Text(d1 ? "d1-2B: one forward pass answers every question"
                        : "Small models, one pass each, on the Neural Engine")
                    .font(.system(size: 24, weight: .bold))
                Text(d1 ? "LiquidAI d1-2B (8-bit) on the GPU: which gesture? which expression? where are the face and the hand? Answers read at their slots, nothing generated."
                        : "\(model.faceDetector.rawValue) face detector → EmotiEffLib EfficientNet-B0 (8 expressions + valence/arousal)  ·  HaGRIDv2 YOLOv10n (33 gestures + no gesture)")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 12)
            // Read-only run status; the settings themselves live in the menu bar (Pipeline menu).
            VStack(alignment: .trailing, spacing: 5) {
                HStack(spacing: 6) {
                    Circle().fill(model.status.hasPrefix("Running") ? Color.green : Color.orange).frame(width: 7, height: 7)
                    Text(model.status).font(Theme.mono)
                }
                Text(configLine)
                    .font(Theme.mono)
                    .foregroundStyle(.secondary)
                Text("Pipeline menu · ⌘1–⌘5 compute · ⌥⌘1/⌥⌘2 face detector · ⌘U uncapped")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }
            .lineLimit(1)
            .help("Change settings in the menu bar: Pipeline")
        }
    }

    private var configLine: String {
        if model.pipeline == .d1 {
            return ["d1-2B", "MLX GPU", "8-bit", model.d1Tokens > 0 ? "\(model.d1Tokens) input tokens" : nil, "⌘K classic"]
                .compactMap { $0 }.joined(separator: " · ")
        }
        var parts = [model.computeMode.rawValue, model.faceDetector.rawValue,
                     model.uncapped ? "uncapped" : "per camera frame"]
        if model.aneWake != .off, model.computeMode != .gpu, model.computeMode != .cpu {
            parts.append(model.aneWake.shortName)
        }
        if model.loadSeconds > 0 { parts.append(String(format: "load %.1f s", model.loadSeconds)) }
        parts.append("⌘D d1")
        return parts.joined(separator: " · ")
    }

    // MARK: - Zoom tiles and cards

    /// Face + two hand slots, always three squares so the row never re-lays out.
    /// `side == nil` lets the tiles share the available width (narrow layout).
    private func tileRow(side: CGFloat?) -> some View {
        HStack(spacing: 12) {
            ZoomTile(title: faceTileTitle, image: model.faceCrop, tint: Theme.face, placeholder: "No face")
                .frame(width: side, height: side)
            // d1 answers one gesture and one hand box, so it gets a single hand slot.
            ForEach(0 ..< (model.pipeline == .d1 ? 1 : 2), id: \.self) { slot in
                let tile = slot < model.handTiles.count ? model.handTiles[slot] : nil
                ZoomTile(title: handTileTitle(slot: slot, hand: tile?.hand), image: tile?.image, tint: Theme.hand,
                         placeholder: slot == 0 ? "No hand" : "No second hand")
                    .frame(width: side, height: side)
            }
        }
    }

    private var faceTileTitle: String {
        guard let face = model.face, model.expressionClasses.indices.contains(face.topIndex) else { return "FACE" }
        let label = Labels.expression(model.expressionClasses[face.topIndex])
        return "FACE · \(label.emoji) \(label.title)"
    }

    private func handTileTitle(slot: Int, hand: HandResult?) -> String {
        let name = model.pipeline == .d1 ? "HAND" : "HAND \(slot + 1)"
        guard let hand, model.gestureClasses.indices.contains(hand.classIndex) else { return name }
        let label = Labels.gesture(model.gestureClasses[hand.classIndex])
        return "\(name) · \(label.emoji) \(label.title)"
    }

    private var cards: some View {
        VStack(spacing: 12) {
            ExpressionCard(model: model)
            GestureCard(model: model)
        }
    }

    // MARK: - Stats

    private var statsBar: some View {
        // Three groups; they wrap as whole groups when the window is narrow.
        let t = model.timings
        return FlowLayout(spacing: 8, lineSpacing: 6) {
            if model.pipeline == .d1 {
                StatGroup(title: "d1 ms", items: [
                    .init("vision", num(model.d1VisionMs)),
                    .init("backbone + readout", num(model.d1BackboneMs)),
                    .init("total", num(model.d1TotalMs), highlight: true),
                ])
                .help("One d1-2B decision on the GPU: vision tower + projector, then the LFM3 backbone over all questions and the answer readout.")
            } else {
                StatGroup(title: "pipeline ms", items: [
                    .init("face", num(t.faceDetect)),
                    .init("expression", num(t.expression)),
                    .init("gestures", num(t.gesture)),
                    .init("total", num(t.total), highlight: true),
                ])
                .help("Per frame. Total also includes downscaling the frame (~2 ms); in Hybrid, gestures overlap face + expression.")
            }
            StatGroup(title: "throughput", items: [
                .init("decisions/s", String(format: "%3.0f", model.inferenceFPS)),
                .init("camera fps", String(format: "%3.0f", model.cameraFPS)),
            ])
            StatGroup(title: "latency ms", items: [
                .init("camera delivery", num(model.cameraDelayMs)),
                .init("capture→decision", num(model.latencyMs)),
            ])
            .help("Camera delivery: capture timestamp → frame arrives in the app (sensor, image processing, driver). Capture→decision adds the pipeline.")
        }
    }

    /// Fixed width in the monospaced font, so groups never change size as numbers change.
    private func num(_ value: Double) -> String { String(format: "%5.1f", min(value, 999.9)) }
}

// MARK: - Components

/// Left-to-right flow that wraps to a new line when the next item doesn't fit.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    var lineSpacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, lineHeight: CGFloat = 0, widest: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > maxWidth {
                x = 0
                y += lineHeight + lineSpacing
                lineHeight = 0
            }
            widest = max(widest, x + size.width)
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
        return CGSize(width: proposal.width ?? widest, height: y + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, lineHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                x = bounds.minX
                y += lineHeight + lineSpacing
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

/// Sizes the video to the column width, but leaves at least `minTile` of height for the zoom row;
/// tiles then take whatever height remains (capped), three across.
struct FeedLayout {
    let feedSize: CGSize
    let tileSide: CGFloat

    init(available: CGSize, frame: CGSize, gap: CGFloat, minTile: CGFloat = 130, maxTile: CGFloat = 280) {
        let aspect = frame.height / max(frame.width, 1)
        var width = available.width
        if width * aspect + gap + minTile > available.height {
            width = max(240, (available.height - gap - minTile) / aspect)
        }
        feedSize = CGSize(width: width, height: width * aspect)
        let byHeight = available.height - feedSize.height - gap
        let byWidth = (width - 2 * gap) / 3
        tileSide = max(60, min(byHeight, byWidth, maxTile))
    }
}

struct ZoomTile: View {
    let title: String
    let image: CGImage?
    let tint: Color
    let placeholder: String

    var body: some View {
        // The square slot alone decides the size; the crop is an overlay, so its changing
        // dimensions can never re-layout the tile or anything below it.
        Color.black.opacity(0.5)
            .aspectRatio(1, contentMode: .fit)
            .frame(maxWidth: .infinity)
            .overlay {
                if let image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .interpolation(.medium)
                        .scaledToFill()
                } else {
                    Text(placeholder)
                        .font(Theme.mono)
                        .foregroundStyle(.tertiary)
                }
            }
            .overlay(alignment: .topLeading) {
                Text(title)
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(.black)
                    .lineLimit(1)
                    .padding(.horizontal, 6).padding(.vertical, 3)
                    .background(tint, in: RoundedRectangle(cornerRadius: 4))
                    .padding(8)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(tint.opacity(image == nil ? 0.15 : 0.6), lineWidth: 1.5))
    }
}

struct Card<Content: View>: View {
    let title: String
    let subtitle: String
    let headline: String
    let tint: Color
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(title).font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundStyle(tint)
                Spacer()
                Text(subtitle).font(Theme.mono).foregroundStyle(.tertiary)
            }
            Text(headline)
                .font(.system(size: 20, weight: .semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

struct ExpressionCard: View {
    let model: AppModel

    var body: some View {
        let scores = model.expressionScores
        let classes = model.expressionClasses
        let top = model.face?.topIndex  // sticky label from the Stabilizer
        let headline: String = {
            guard let top, classes.indices.contains(top), top < scores.count else { return "No face" }
            let l = Labels.expression(classes[top])
            return "\(l.emoji) \(l.title) \(percent(scores[top]))"
        }()

        let d1 = model.pipeline == .d1
        return Card(title: "FACE · EXPRESSION", subtitle: d1 ? "d1-2B · calibrated" : "EmotiEffLib B0 · softmax",
                    headline: headline, tint: Theme.face) {
            VStack(spacing: 5) {
                ForEach(Array(classes.enumerated()), id: \.offset) { index, name in
                    let label = Labels.expression(name)
                    let value = index < scores.count ? scores[index] : 0
                    BarRow(emoji: label.emoji, title: label.title, value: value, isTop: index == top, tint: Theme.face)
                }
            }
            if !d1 {  // valence / arousal come from EmotiEffLib only
                Divider().opacity(0.3)
                CenteredBar(title: "Valence", value: model.valence, tint: Theme.face)
                CenteredBar(title: "Arousal", value: model.arousal, tint: Theme.face)
            }
        }
    }
}

struct GestureCard: View {
    let model: AppModel

    var body: some View {
        let scores = model.gestureScores
        let classes = model.gestureClasses
        let ranked = model.gestureRows.filter { scores.indices.contains($0) }  // sticky order
        let primaryClass = model.hands.first?.classIndex
        let rows = model.gestureRowCount
        let headline: String = {
            guard let hand = model.hands.first, classes.indices.contains(hand.classIndex) else { return "No hand" }
            let l = Labels.gesture(classes[hand.classIndex])
            return "\(l.emoji) \(l.title) \(percent(hand.score))"
        }()

        let subtitle = model.pipeline == .d1 ? "d1-2B · \(classes.count) options · calibrated"
                                             : "HaGRIDv2 YOLOv10n · top \(rows) of \(classes.count)"
        return Card(title: "HAND · GESTURE", subtitle: subtitle, headline: headline, tint: Theme.hand) {
            VStack(spacing: 5) {
                ForEach(ranked, id: \.self) { index in
                    let label = Labels.gesture(classes.indices.contains(index) ? classes[index] : "?")
                    BarRow(emoji: label.emoji, title: label.title, value: scores[index],
                           isTop: index == primaryClass, tint: Theme.hand)
                }
                if ranked.isEmpty {
                    Text("Show a hand to the camera").font(Theme.mono).foregroundStyle(.tertiary)
                }
            }
            // Fixed height for all rows, so the card never grows or shrinks.
            .frame(height: CGFloat(rows) * 17 + CGFloat(rows - 1) * 5, alignment: .top)
            .animation(.easeInOut(duration: 0.25), value: ranked)
            Text("\(max(model.hands.count, 2)) hands detected · showing the most confident")
                .font(Theme.mono).foregroundStyle(.tertiary)
                .opacity(model.hands.count > 1 ? 1 : 0)  // line space stays reserved
        }
    }
}

struct BarRow: View {
    let emoji: String
    let title: String
    let value: Float
    let isTop: Bool
    let tint: Color

    var body: some View {
        HStack(spacing: 8) {
            Text(emoji).font(.system(size: 13)).frame(width: 20)
            Text(title)
                .font(.system(size: 12, weight: isTop ? .semibold : .regular))
                .foregroundStyle(isTop ? .primary : .secondary)
                .frame(width: 120, alignment: .leading)
                .lineLimit(1)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(.white.opacity(0.07))
                    Capsule().fill(tint.opacity(isTop ? 1 : 0.55))
                        .frame(width: max(2, geo.size.width * CGFloat(min(max(value, 0), 1))))
                }
            }
            .frame(height: 6)
            Text(percent(value))
                .font(Theme.mono)
                .foregroundStyle(isTop ? .primary : .secondary)
                .frame(width: 38, alignment: .trailing)
        }
        .frame(height: 17)
    }
}

/// Bar for a value in -1...1, filled from the center.
struct CenteredBar: View {
    let title: String
    let value: Float
    let tint: Color

    var body: some View {
        HStack(spacing: 8) {
            Text(title).font(.system(size: 12)).foregroundStyle(.secondary).frame(width: 148, alignment: .leading)
            GeometryReader { geo in
                let half = geo.size.width / 2
                let v = CGFloat(min(max(value, -1), 1))
                ZStack(alignment: .leading) {
                    Capsule().fill(.white.opacity(0.07))
                    Rectangle().fill(.white.opacity(0.25)).frame(width: 1).offset(x: half)
                    Capsule().fill(tint)
                        .frame(width: max(2, abs(v) * half))
                        .offset(x: v >= 0 ? half : half + v * half)
                }
            }
            .frame(height: 6)
            Text(String(format: "%+.2f", value))
                .font(Theme.mono)
                .foregroundStyle(.secondary)
                .frame(width: 38, alignment: .trailing)
        }
        .frame(height: 17)
    }
}

/// A capsule holding several label/value pairs, e.g. "pipeline ms  face 3.2 │ expression 2.4 │ …".
struct StatGroup: View {
    struct Item {
        var label: String
        var value: String
        var highlight = false

        init(_ label: String, _ value: String, highlight: Bool = false) {
            self.label = label
            self.value = value
            self.highlight = highlight
        }
    }

    let title: String
    let items: [Item]

    var body: some View {
        HStack(spacing: 8) {
            Text(title).foregroundStyle(.tertiary)
            ForEach(items.indices, id: \.self) { index in
                if index > 0 {
                    Rectangle().fill(.white.opacity(0.12)).frame(width: 1, height: 11)
                }
                HStack(spacing: 4) {
                    Text(items[index].label).foregroundStyle(.secondary)
                    Text(items[index].value).foregroundStyle(items[index].highlight ? Theme.hand : .primary)
                }
            }
        }
        .font(Theme.mono)
        .lineLimit(1)
        .fixedSize()
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Theme.card, in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.08)))
    }
}

func percent(_ value: Float) -> String { "\(Int((min(max(value, 0), 1) * 100).rounded()))%" }
