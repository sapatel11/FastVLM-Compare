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
        let ttft = model.timeToFirstToken
        let total = model.totalLatency
        let tokens = model.generatedTokenCount
        let tokensPerSecond = model.tokensPerSecond

        guard !output.isEmpty, !output.hasPrefix("Failed:") else {
            throw RuntimeProbeError.generationFailed(variant, model.output)
        }
        guard ttft.isFinite, ttft > 0 else {
            throw RuntimeProbeError.invalidMetric(variant, "TTFT", ttft)
        }
        guard total.isFinite, total > 0, total >= ttft else {
            throw RuntimeProbeError.invalidMetric(variant, "total latency", total)
        }
        guard tokens > 0 else {
            throw RuntimeProbeError.invalidTokenCount(variant, tokens)
        }
        guard tokensPerSecond.isFinite, tokensPerSecond > 0 else {
            throw RuntimeProbeError.invalidMetric(variant, "tokens/sec", tokensPerSecond)
        }

        let singleLineOutput = output.replacingOccurrences(of: "\n", with: " ")
        print("FASTVLM_RUNTIME_PROBE PASS \(variant.rawValue)")
        print(String(format: "FASTVLM_RUNTIME_PROBE TTFT %@: %.3f ms", variant.rawValue, ttft * 1000))
        print(String(format: "FASTVLM_RUNTIME_PROBE TOTAL %@: %.3f ms", variant.rawValue, total * 1000))
        print("FASTVLM_RUNTIME_PROBE TOKENS \(variant.rawValue): \(tokens)")
        print(String(format: "FASTVLM_RUNTIME_PROBE TOKENS_PER_SECOND %@: %.3f", variant.rawValue, tokensPerSecond))
        print("FASTVLM_RUNTIME_PROBE OUTPUT \(variant.rawValue): \(singleLineOutput)")
    }

    private enum RuntimeProbeError: LocalizedError {
        case missingImagePath
        case invalidImage(String)
        case generationFailed(FastVLMVariant, String)
        case invalidMetric(FastVLMVariant, String, Double)
        case invalidTokenCount(FastVLMVariant, Int)

        var errorDescription: String? {
            switch self {
            case .missingImagePath:
                return "FASTVLM_RUNTIME_PROBE_IMAGE was not provided."
            case .invalidImage(let path):
                return "Could not decode the runtime probe image at \(path)."
            case .generationFailed(let variant, let output):
                return "\(variant.rawValue) native generation failed: \(output)"
            case .invalidMetric(let variant, let metric, let value):
                return "\(variant.rawValue) native generation produced invalid \(metric): \(value)."
            case .invalidTokenCount(let variant, let value):
                return "\(variant.rawValue) native generation produced invalid token count: \(value)."
            }
        }
    }
}
#endif
