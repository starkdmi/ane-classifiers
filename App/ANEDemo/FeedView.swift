import AVFoundation
import SwiftUI

/// Hosts the shared AVSampleBufferDisplayLayer (frames are enqueued straight from the camera queue).
struct FeedView: NSViewRepresentable {
    let displayLayer: AVSampleBufferDisplayLayer

    func makeNSView(context: Context) -> LayerHostView { LayerHostView(hosted: displayLayer) }
    func updateNSView(_ nsView: LayerHostView, context: Context) {}
}

final class LayerHostView: NSView {
    private let hosted: CALayer

    init(hosted: CALayer) {
        self.hosted = hosted
        super.init(frame: .zero)
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.cgColor
        layer?.addSublayer(hosted)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    override func layout() {
        super.layout()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        hosted.frame = bounds
        CATransaction.commit()
    }
}

/// Live feed with face / hand boxes drawn on top, in the same normalized top-left space.
/// One labeled box on the video, in normalized (0...1, top-left) frame coordinates.
struct OverlayBox: Identifiable, Equatable {
    var id: String
    var rect: CGRect
    var color: Color
    var text: String
}

struct AnnotatedFeed: View {
    let model: AppModel

    /// Boxes as views (not a Canvas) so their frames can animate between d1's decisions.
    private var boxes: [OverlayBox] {
        var boxes: [OverlayBox] = []
        for (index, hand) in model.hands.enumerated() {
            let key = model.gestureClasses.indices.contains(hand.classIndex) ? model.gestureClasses[hand.classIndex] : "?"
            let label = Labels.gesture(key)
            boxes.append(OverlayBox(id: "hand\(index)", rect: hand.rect,
                                    color: index == 0 ? Theme.hand : Theme.hand.opacity(0.55),
                                    text: "\(label.emoji) \(label.title) \(Int((hand.score * 100).rounded()))%"))
        }
        if let face = model.face, model.expressionClasses.indices.contains(face.topIndex) {
            let label = Labels.expression(model.expressionClasses[face.topIndex])
            boxes.append(OverlayBox(id: "face", rect: face.rect, color: Theme.face,
                                    text: "\(label.emoji) \(label.title) \(Int((face.probs[face.topIndex] * 100).rounded()))%"))
        }
        // d1 answered but found no box (or the answer is "none"): nothing to draw.
        return boxes.filter { $0.rect.width > 0.001 && $0.rect.height > 0.001 }
    }

    var body: some View {
        ZStack {
            FeedView(displayLayer: model.displayLayer)
            GeometryReader { geo in
                ZStack(alignment: .topLeading) {
                    ForEach(boxes) { box in
                        BoxView(box: box, size: geo.size)
                    }
                }
                .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
                // d1 updates ~2x a second: glide between its boxes (0.12 s, like the mlx-d1-gestures web demo).
                // The classic pipeline updates every frame, so its boxes move instantly.
                .animation(model.pipeline == .d1 ? .linear(duration: 0.12) : nil, value: boxes)
            }
            .allowsHitTesting(false)
        }
        .aspectRatio(model.frameSize.width / max(model.frameSize.height, 1), contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(.white.opacity(0.08)))
    }
}

private struct BoxView: View {
    let box: OverlayBox
    let size: CGSize

    var body: some View {
        let r = CGRect(x: box.rect.minX * size.width, y: box.rect.minY * size.height,
                       width: box.rect.width * size.width, height: box.rect.height * size.height)
        let tagHeight: CGFloat = 23
        RoundedRectangle(cornerRadius: 6)
            .stroke(box.color, lineWidth: 2.5)
            .frame(width: r.width, height: r.height)
            .overlay(alignment: .topLeading) {
                Text(box.text)
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.black)
                    .lineLimit(1)
                    .fixedSize()
                    .padding(.horizontal, 6)
                    .frame(height: tagHeight)
                    .background(box.color, in: RoundedRectangle(cornerRadius: 5))
                    // above the box, or inside it when the box touches the top edge
                    .offset(y: r.minY >= tagHeight + 2 ? -(tagHeight + 2) : 0)
            }
            .position(x: r.midX, y: r.midY)
    }
}
