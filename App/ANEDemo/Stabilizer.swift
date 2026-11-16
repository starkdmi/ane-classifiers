import CoreGraphics
import Foundation

/// One Euro filter (Casiez et al., 2012): strong smoothing while a value is still, little lag
/// while it moves. `minCutoff` (Hz) sets the still-time jitter; `beta` how fast it opens up with speed.
struct OneEuroFilter {
    let minCutoff: Double
    let beta: Double
    let derivativeCutoff = 1.0
    private var value: Double?
    private var derivative = 0.0
    private var lastTime: Double?

    init(minCutoff: Double, beta: Double) {
        self.minCutoff = minCutoff
        self.beta = beta
    }

    mutating func filter(_ x: Double, time: Double) -> Double {
        guard let previous = value, let lastTime else {
            value = x
            self.lastTime = time
            return x
        }
        let dt = max(time - lastTime, 1e-3)
        derivative += Self.alpha(dt: dt, cutoff: derivativeCutoff) * ((x - previous) / dt - derivative)
        let cutoff = minCutoff + beta * abs(derivative)
        let result = previous + Self.alpha(dt: dt, cutoff: cutoff) * (x - previous)
        value = result
        self.lastTime = time
        return result
    }

    private static func alpha(dt: Double, cutoff: Double) -> Double {
        1 / (1 + 1 / (2 * Double.pi * cutoff) / dt)
    }
}

/// Smooths a normalized rect as center + size. Size is smoothed harder than position, so crops and
/// zoom tiles don't "breathe" when the detector's box size jitters.
struct RectFilter {
    private var cx = OneEuroFilter(minCutoff: 0.5, beta: 6)
    private var cy = OneEuroFilter(minCutoff: 0.5, beta: 6)
    private var w = OneEuroFilter(minCutoff: 0.2, beta: 2)
    private var h = OneEuroFilter(minCutoff: 0.2, beta: 2)

    mutating func filter(_ r: CGRect, time: Double) -> CGRect {
        let x = cx.filter(r.midX, time: time), y = cy.filter(r.midY, time: time)
        let width = w.filter(r.width, time: time), height = h.filter(r.height, time: time)
        return CGRect(x: x - width / 2, y: y - height / 2, width: width, height: height)
    }
}

enum ScoreMath {
    static func ema(_ current: [Float], toward target: [Float], alpha: Float) -> [Float] {
        guard current.count == target.count else { return target }
        return zip(current, target).map { $0 + ($1 - $0) * alpha }
    }

    static func argmax(_ scores: [Float]) -> Int {
        scores.indices.max { scores[$0] < scores[$1] } ?? 0
    }

    /// Keeps `current` unless another class beats it by more than `margin`.
    static func stickyArgmax(_ scores: [Float], current: Int?, margin: Float) -> Int {
        let best = argmax(scores)
        guard let current, scores.indices.contains(current) else { return best }
        return scores[best] > scores[current] + margin ? best : current
    }

    /// Top-`count` rows that only change membership or order when a row clearly overtakes another.
    static func stickyTopRows(_ scores: [Float], previous: [Int], count: Int, margin: Float) -> [Int] {
        guard !scores.isEmpty else { return [] }
        let ranked = scores.indices.sorted { scores[$0] > scores[$1] }
        var rows = previous.filter { scores.indices.contains($0) }
        for i in ranked where rows.count < count && !rows.contains(i) { rows.append(i) }

        // Outsiders (strongest first) replace the weakest shown row only with a clear lead.
        for i in ranked where !rows.contains(i) {
            guard let weakest = rows.indices.min(by: { scores[rows[$0]] < scores[rows[$1]] }),
                  scores[i] > scores[rows[weakest]] + margin else { break }
            rows[weakest] = i
        }

        // Bubble rows up only when they beat the row above by the margin.
        var swapped = true
        while swapped && rows.count > 1 {
            swapped = false
            for k in 1..<rows.count where scores[rows[k]] > scores[rows[k - 1]] + margin {
                rows.swapAt(k, k - 1)
                swapped = true
            }
        }
        return rows
    }
}

/// Turns raw per-frame detections into steady ones: smoothed boxes, smoothed scores, sticky labels,
/// short hold-over when a detection drops for a frame, and per-hand tracks matched by IoU.
/// Used only on the inference queue.
final class Stabilizer {
    private let scoreAlpha: Float = 0.3
    private let labelMargin: Float = 0.12
    private let faceHold = 0.3
    private let handHold = 0.25
    private let minHandHits = 2

    private var faceFilter = RectFilter()
    private var face: FaceResult?
    private var faceLastSeen = 0.0

    private struct Track {
        let id: Int
        var filter: RectFilter
        var hand: HandResult
        var lastSeen: Double
        var hits: Int
    }
    private var tracks: [Track] = []
    private var nextTrackID = 0
    private var primaryID: Int?

    func reset() {
        faceFilter = RectFilter()
        face = nil
        tracks = []
        primaryID = nil
    }

    func update(face raw: FaceResult?, hands rawHands: [HandResult], time: Double) -> (FaceResult?, [HandResult]) {
        updateFace(raw, time: time)
        updateHands(rawHands, time: time)

        let visible = tracks.filter { $0.hits >= minHandHits }
        if primaryID == nil || !visible.contains(where: { $0.id == primaryID }) {
            primaryID = visible.max { $0.hand.score < $1.hand.score }?.id
        }
        let ordered = visible.sorted {
            ($0.id == primaryID ? 1 : 0, $0.hand.score) > ($1.id == primaryID ? 1 : 0, $1.hand.score)
        }
        return (face, ordered.map(\.hand))
    }

    private func updateFace(_ raw: FaceResult?, time: Double) {
        guard let raw else {
            if face != nil, time - faceLastSeen > faceHold {
                face = nil
                faceFilter = RectFilter()
            }
            return
        }
        var next = raw
        next.rect = faceFilter.filter(raw.rect, time: time)
        if let previous = face {
            next.probs = ScoreMath.ema(previous.probs, toward: raw.probs, alpha: scoreAlpha)
            next.valence = previous.valence + (raw.valence - previous.valence) * scoreAlpha
            next.arousal = previous.arousal + (raw.arousal - previous.arousal) * scoreAlpha
            next.topIndex = ScoreMath.stickyArgmax(next.probs, current: previous.topIndex, margin: labelMargin)
        }
        face = next
        faceLastSeen = time
    }

    private func updateHands(_ rawHands: [HandResult], time: Double) {
        var claimed = Set<Int>()
        for raw in rawHands.sorted(by: { $0.score > $1.score }) {
            let match = tracks.indices
                .filter { !claimed.contains(tracks[$0].id) }
                .max { tracks[$0].hand.rect.iou(raw.rect) < tracks[$1].hand.rect.iou(raw.rect) }
            if let t = match, tracks[t].hand.rect.iou(raw.rect) > 0.2 {
                var track = tracks[t]
                var hand = raw
                hand.rect = track.filter.filter(raw.rect, time: time)
                hand.scores = ScoreMath.ema(track.hand.scores, toward: raw.scores, alpha: scoreAlpha)
                hand.classIndex = ScoreMath.stickyArgmax(hand.scores, current: track.hand.classIndex, margin: labelMargin)
                hand.score = hand.scores[hand.classIndex]
                track.hand = hand
                track.lastSeen = time
                track.hits += 1
                tracks[t] = track
                claimed.insert(track.id)
            } else {
                var filter = RectFilter()
                var hand = raw
                hand.rect = filter.filter(raw.rect, time: time)
                tracks.append(Track(id: nextTrackID, filter: filter, hand: hand, lastSeen: time, hits: 1))
                claimed.insert(nextTrackID)
                nextTrackID += 1
            }
        }
        tracks.removeAll { time - $0.lastSeen > handHold }
    }
}
