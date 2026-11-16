import Foundation
import QuartzCore

/// Collects per-step timings and summarizes them as median / p95 / max over a time window.
/// Used only on the inference queue.
final class Profiler {
    private var samples: [String: [Double]] = [:]
    private var order: [String] = []
    private var windowStart = CACurrentMediaTime()

    func record(_ name: String, _ ms: Double) {
        if samples[name] == nil { order.append(name) }
        samples[name, default: []].append(ms)
    }

    /// Times `body` and records it under `name`.
    func measure<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        let start = CACurrentMediaTime()
        defer { record(name, (CACurrentMediaTime() - start) * 1000) }
        return try body()
    }

    /// Returns a one-line summary and starts a new window once `interval` seconds have passed.
    func reportIfDue(interval: Double = 3) -> String? {
        let now = CACurrentMediaTime()
        guard now - windowStart >= interval, !order.isEmpty else { return nil }
        let frames = samples["total"]?.count ?? 0
        let parts = order.compactMap { name -> String? in
            guard let values = samples[name]?.sorted(), !values.isEmpty else { return nil }
            func pct(_ p: Double) -> Double { values[min(values.count - 1, Int(Double(values.count - 1) * p + 0.5))] }
            return String(format: "%@ %.1f/%.1f/%.1f", name, pct(0.5), pct(0.95), values.last!)
        }
        let rate = Double(frames) / (now - windowStart)
        samples.removeAll(keepingCapacity: true)
        order.removeAll(keepingCapacity: true)
        windowStart = now
        return String(format: "%.1f runs/s · p50/p95/max ms: ", rate) + parts.joined(separator: " | ")
    }
}
