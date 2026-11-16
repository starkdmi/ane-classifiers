// Lists every video camera with its formats (dimensions, pixel format, frame-rate ranges).
// Usage: swift tools/camera_formats.swift
import AVFoundation

let devices = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
    mediaType: .video, position: .unspecified
).devices

for device in devices {
    print("== \(device.localizedName) [\(device.deviceType.rawValue)]")
    let active = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
    print("   active: \(active.width)x\(active.height)")
    for format in device.formats {
        let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let sub = CMFormatDescriptionGetMediaSubType(format.formatDescription)
        let fourcc = String(bytes: [24, 16, 8, 0].map { UInt8((sub >> $0) & 0xff) }, encoding: .ascii) ?? "?"
        let rates = format.videoSupportedFrameRateRanges
            .map { String(format: "%.0f-%.0f", $0.minFrameRate, $0.maxFrameRate) }
            .joined(separator: ", ")
        print("   \(d.width)x\(d.height) \(fourcc) fps: \(rates)")
    }
}
