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
    private let benchmarkEnabled =
        ProcessInfo.processInfo.environment["FASTVLM_BENCHMARK"] == "1"

    var body: some Scene {
        WindowGroup {
            #if os(macOS)
            if benchmarkEnabled {
                FastVLMBenchmarkView()
            } else if runtimeProbeEnabled {
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

private struct FastVLMBenchmarkView: View {
    @State private var model = FastVLMModel()

    var body: some View {
        Text("Running FastVLM benchmark…")
            .task {
                await runBenchmark()
            }
    }

    @MainActor
    private func runBenchmark() async {
        do {
            let environment = ProcessInfo.processInfo.environment
            guard let manifestPath = environment["FASTVLM_BENCHMARK_MANIFEST"],
                  !manifestPath.isEmpty else {
                throw BenchmarkError.missingManifestPath
            }
            guard let outputPath = environment["FASTVLM_BENCHMARK_OUTPUT"],
                  !outputPath.isEmpty else {
                throw BenchmarkError.missingOutputPath
            }

            let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
            let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
            let manifestData = try Data(contentsOf: manifestURL)
            let entries = try JSONDecoder().decode([BenchmarkManifestEntry].self, from: manifestData)

            guard !entries.isEmpty else {
                throw BenchmarkError.emptyManifest
            }

            var records: [BenchmarkRecord] = []
            records.reserveCapacity(entries.count * FastVLMVariant.allCases.count)

            for entry in entries {
                let imageURL = URL(
                    fileURLWithPath: entry.imagePath,
                    relativeTo: manifestURL.deletingLastPathComponent()
                ).standardizedFileURL
                let imageData = try Data(contentsOf: imageURL)
                guard let image = CIImage(data: imageData) else {
                    throw BenchmarkError.invalidImage(entry.imageID, imageURL.path)
                }

                for variant in FastVLMVariant.allCases {
                    model.selectedVariant = variant

                    let userInput = UserInput(
                        prompt: .text(entry.prompt),
                        images: [.ciImage(image)]
                    )

                    guard let result = await model.generateResult(userInput) else {
                        throw BenchmarkError.generationFailed(entry.imageID, variant, model.output)
                    }

                    let record = BenchmarkRecord(
                        imageID: entry.imageID,
                        variant: variant.rawValue,
                        caption: result.caption,
                        ttftMS: result.timeToFirstToken * 1000,
                        totalLatencyMS: result.totalLatency * 1000,
                        generatedTokens: result.generatedTokenCount,
                        tokensPerSecond: result.tokensPerSecond
                    )
                    records.append(record)

                    print(
                        String(
                            format: "FASTVLM_BENCHMARK PASS %@ %@ TTFT=%.3fms TOTAL=%.3fms TOKENS=%d TPS=%.3f",
                            entry.imageID,
                            variant.rawValue,
                            record.ttftMS,
                            record.totalLatencyMS,
                            record.generatedTokens,
                            record.tokensPerSecond
                        )
                    )
                }
            }

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let jsonLines = try records.map { record -> String in
                let data = try encoder.encode(record)
                guard let line = String(data: data, encoding: .utf8) else {
                    throw BenchmarkError.encodingFailed
                }
                return line
            }

            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try (jsonLines.joined(separator: "\n") + "\n").write(
                to: outputURL,
                atomically: true,
                encoding: .utf8
            )

            print("FASTVLM_BENCHMARK PASS all records=\(records.count)")
            print("FASTVLM_BENCHMARK OUTPUT \(outputURL.path)")
            Darwin.exit(EXIT_SUCCESS)
        } catch {
            print("FASTVLM_BENCHMARK FAIL: \(error.localizedDescription)")
            Darwin.exit(EXIT_FAILURE)
        }
    }

    private struct BenchmarkManifestEntry: Decodable {
        let imageID: String
        let imagePath: String
        let prompt: String

        enum CodingKeys: String, CodingKey {
            case imageID = "image_id"
            case imagePath = "image_path"
            case prompt
        }
    }

    private struct BenchmarkRecord: Encodable {
        let imageID: String
        let variant: String
        let caption: String
        let ttftMS: Double
        let totalLatencyMS: Double
        let generatedTokens: Int
        let tokensPerSecond: Double

        enum CodingKeys: String, CodingKey {
            case imageID = "image_id"
            case variant
            case caption
            case ttftMS = "TTFT_ms"
            case totalLatencyMS = "total_latency_ms"
            case generatedTokens = "generated_tokens"
            case tokensPerSecond = "tokens_per_second"
        }
    }

    private enum BenchmarkError: LocalizedError {
        case missingManifestPath
        case missingOutputPath
        case emptyManifest
        case invalidImage(String, String)
        case generationFailed(String, FastVLMVariant, String)
        case encodingFailed

        var errorDescription: String? {
            switch self {
            case .missingManifestPath:
                return "FASTVLM_BENCHMARK_MANIFEST was not provided."
            case .missingOutputPath:
                return "FASTVLM_BENCHMARK_OUTPUT was not provided."
            case .emptyManifest:
                return "The benchmark manifest contains no entries."
            case .invalidImage(let imageID, let path):
                return "Benchmark image \(imageID) could not be decoded at \(path)."
            case .generationFailed(let imageID, let variant, let output):
                return "Benchmark generation failed for \(imageID) with \(variant.rawValue): \(output)"
            case .encodingFailed:
                return "Could not encode a benchmark JSONL record as UTF-8."
            }
        }
    }
}
#endif
