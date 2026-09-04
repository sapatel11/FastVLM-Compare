//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import SwiftUI

#if os(macOS)
import CoreImage
import Darwin
import Foundation
import MLXLMCommon
#endif

@main
struct FastVLMApp: App {
    private let runtimeProbeEnabled =
        ProcessInfo.processInfo.environment["FASTVLM_RUNTIME_PROBE"] == "1"

    var body: some Scene {
        WindowGroup {
            #if os(macOS)
            if runtimeProbeEnabled {
                FastVLMRuntimeProbeView()
            } else {
                ContentView()
            }
            #else
            ContentView()
            #endif
        }
    }
}

#if os(macOS)
private struct FastVLMRuntimeProbeView: View {
    @State private var model = FastVLMModel()

    var body: some View {
        Text("Running FastVLM native runtime probe…")
            .task {
                await runProbe()
            }
    }

    @MainActor
    private func runProbe() async {
        do {
            let environment = ProcessInfo.processInfo.environment
            guard let imagePath = environment["FASTVLM_RUNTIME_PROBE_IMAGE"],
                  !imagePath.isEmpty else {
                throw RuntimeProbeError.missingImagePath
            }

            let imageData = try Data(contentsOf: URL(fileURLWithPath: imagePath))
            guard let image = CIImage(data: imageData) else {
                throw RuntimeProbeError.invalidImage(imagePath)
            }

            try await runVariant(.int8, image: image)
            try await runVariant(.int4, image: image)

            print("FASTVLM_RUNTIME_PROBE PASS all")
            Darwin.exit(EXIT_SUCCESS)
        } catch {
            print("FASTVLM_RUNTIME_PROBE FAIL: \(error.localizedDescription)")
            Darwin.exit(EXIT_FAILURE)
        }
    }

    @MainActor
    private func runVariant(_ variant: FastVLMVariant, image: CIImage) async throws {
        model.selectedVariant = variant

        let userInput = UserInput(
            prompt: .text("Describe this image briefly."),
            images: [.ciImage(image)]
        )

        let task = await model.generate(userInput)
        _ = await task.result

        let output = model.output.trimmingCharacters(in: .whitespacesAndNewlines)
        let ttft = model.promptTime.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !output.isEmpty, !output.hasPrefix("Failed:") else {
            throw RuntimeProbeError.generationFailed(variant, model.output)
        }
        guard !ttft.isEmpty else {
            throw RuntimeProbeError.missingTTFT(variant)
        }

        let singleLineOutput = output.replacingOccurrences(of: "\n", with: " ")
        print("FASTVLM_RUNTIME_PROBE PASS \(variant.rawValue)")
        print("FASTVLM_RUNTIME_PROBE TTFT \(variant.rawValue): \(ttft)")
        print("FASTVLM_RUNTIME_PROBE OUTPUT \(variant.rawValue): \(singleLineOutput)")
    }

    private enum RuntimeProbeError: LocalizedError {
        case missingImagePath
        case invalidImage(String)
        case generationFailed(FastVLMVariant, String)
        case missingTTFT(FastVLMVariant)

        var errorDescription: String? {
            switch self {
            case .missingImagePath:
                return "FASTVLM_RUNTIME_PROBE_IMAGE was not provided."
            case .invalidImage(let path):
                return "Could not decode the runtime probe image at \(path)."
            case .generationFailed(let variant, let output):
                return "\(variant.rawValue) native generation failed: \(output)"
            case .missingTTFT(let variant):
                return "\(variant.rawValue) native generation produced no TTFT."
            }
        }
    }
}
#endif
