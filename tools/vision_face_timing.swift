// Times VNDetectFaceRectanglesRequest on a 640x384 frame per compute device (15 runs each, light).
// Usage: swift tools/vision_face_timing.swift weights/testdata/example.jpeg
import AppKit
import CoreML
import Vision

let path = CommandLine.arguments[1]
guard let image = NSImage(contentsOfFile: path),
      let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { fatalError("no image") }

// Draw the image into a 640x384 BGRA pixel buffer (like the app's letterboxed frame).
var pb: CVPixelBuffer?
CVPixelBufferCreate(nil, 640, 384, kCVPixelFormatType_32BGRA,
                    [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
let buffer = pb!
CVPixelBufferLockBaseAddress(buffer, [])
let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: 640, height: 384, bitsPerComponent: 8,
                    bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
ctx.draw(cg, in: CGRect(x: 0, y: 32, width: 640, height: 320))
CVPixelBufferUnlockBaseAddress(buffer, [])

func time(_ label: String, configure: (VNDetectFaceRectanglesRequest) -> Void) {
    let request = VNDetectFaceRectanglesRequest()
    configure(request)
    let handler = { try! VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request]) }
    handler(); handler()  // warm-up
    var samples: [Double] = []
    for _ in 0..<15 {
        let t = CACurrentMediaTime()
        handler()
        samples.append((CACurrentMediaTime() - t) * 1000)
    }
    samples.sort()
    print(String(format: "%-14@ median %.1f ms  min %.1f ms  faces %d", label as NSString, samples[7], samples[0], request.results?.count ?? 0))
}

let devices = (try? VNDetectFaceRectanglesRequest().supportedComputeStageDevices[.main]) ?? []
print("devices:", devices.map { "\($0)" })
print("revisions:", Array(VNDetectFaceRectanglesRequest.supportedRevisions))
for revision in VNDetectFaceRectanglesRequest.supportedRevisions {
    time("rev \(revision)") { $0.revision = revision }
}
time("default") { _ in }
for device in devices {
    time(String(describing: device).components(separatedBy: "(").first ?? "?") { $0.setComputeDevice(device, for: .main) }
}
