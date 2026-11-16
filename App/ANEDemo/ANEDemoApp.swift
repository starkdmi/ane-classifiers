import SwiftUI

@main
struct ANEDemoApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup("ANE Classifiers") {
            ContentView(model: model)
                .onAppear { model.start() }
        }
        .defaultSize(width: 1360, height: 840)
        .windowToolbarStyle(.unifiedCompact)
        .commands { PipelineCommands(model: model) }
    }
}

/// Menu-bar "Pipeline" menu: all run settings live here (with shortcuts for live switching),
/// keeping the window itself free of controls.
struct PipelineCommands: Commands {
    let model: AppModel

    var body: some Commands {
        CommandMenu("Pipeline") {
            Section("Model") {
                Toggle(PipelineKind.classic.title, isOn: Binding(get: { model.pipeline == .classic },
                                                                 set: { if $0 { model.setPipeline(.classic) } }))
                    .keyboardShortcut("k", modifiers: .command)
                Toggle(PipelineKind.d1.title, isOn: Binding(get: { model.pipeline == .d1 },
                                                            set: { if $0 { model.setPipeline(.d1) } }))
                    .keyboardShortcut("d", modifiers: .command)
                Button("Locate d1 Weights…") { model.locateD1Weights() }
            }
            // Everything below tunes the classic (Core ML) pipeline only.
            let classicOff = model.pipeline != .classic
            Section("Compute (classic)") {
                ForEach(Array(ComputeMode.allCases.enumerated()), id: \.element) { index, mode in
                    Toggle(mode.title, isOn: Binding(get: { model.computeMode == mode },
                                                     set: { if $0 { model.setComputeMode(mode) } }))
                        .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
                        .disabled(classicOff)
                }
            }
            Section("Face detector (classic)") {
                ForEach(Array(FaceDetectorKind.allCases.enumerated()), id: \.element) { index, kind in
                    Toggle(kind.title, isOn: Binding(get: { model.faceDetector == kind },
                                                     set: { if $0 { model.setFaceDetector(kind) } }))
                        .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: [.command, .option])
                        .disabled(classicOff)
                }
            }
            Section("ANE wake-up (classic)") {
                ForEach(ANEWakeMode.allCases) { mode in
                    Toggle(mode.title, isOn: Binding(get: { model.aneWake == mode },
                                                     set: { if $0 { model.setANEWake(mode) } }))
                        .disabled(classicOff)
                }
            }
            Section("Run") {
                Toggle("Uncapped: re-run on the newest frame (classic)", isOn: Binding(get: { model.uncapped },
                                                                                        set: { model.setUncapped($0) }))
                    .keyboardShortcut("u", modifiers: .command)
                    .disabled(classicOff)
                Toggle(model.cameraMaxFPS > 30.5 ? "Camera at \(Int(model.cameraMaxFPS.rounded())) fps"
                                                 : "Camera above 30 fps (not supported by this camera)",
                       isOn: Binding(get: { model.highFrameRate }, set: { model.setHighFrameRate($0) }))
                    .disabled(model.cameraMaxFPS <= 30.5)
            }
            if model.cameras.count > 1 {
                Section("Camera") {
                    ForEach(model.cameras) { camera in
                        Toggle(camera.name, isOn: Binding(get: { model.selectedCameraID == camera.id },
                                                          set: { if $0 { model.selectCamera(camera.id) } }))
                    }
                }
            }
        }
    }
}
