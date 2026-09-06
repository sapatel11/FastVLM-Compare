//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

#if os(macOS)
import CoreImage
import Darwin
import Foundation
import ImageIO
import MLXLMCommon

@MainActor
enum FastVLMBenchmarkRunner {
    static func run() async {
        let model = FastVLMModel()

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

            let variants = try benchmarkVariants(environment)
            let warmupRuns = try benchmarkWarmupRuns(environment)

            let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
            let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
            trace("manifest=\(manifestURL.path) output=\(outputURL.path)")
            trace("max_tokens=\(model.effectiveGenerationTokenLimit)")
            trace(
                "variants=\(variants.map(\.rawValue).joined(separator: ",")) warmup_runs=\(warmupRuns)"
            )

            let manifestData = try Data(contentsOf: manifestURL)
            let entries = try JSONDecoder().decode([BenchmarkManifestEntry].self, from: manifestData)
            trace("manifest decoded entries=\(entries.count)")

            guard !entries.isEmpty else {
                throw BenchmarkError.emptyManifest
            }

            if warmupRuns > 0 {
                let warmupEntry = entries[0]
                let warmupImage = try loadImage(
                    for: warmupEntry,
                    relativeTo: manifestURL.deletingLastPathComponent()
                )

                for variant in variants {
                    model.selectedVariant = variant

                    for warmupIndex in 1...warmupRuns {
                        trace(
                            "warmup start index=\(warmupIndex)/\(warmupRuns) image_id=\(warmupEntry.imageID) variant=\(variant.rawValue)"
                        )

                        let userInput = UserInput(
                            prompt: .text(warmupEntry.prompt),
                            images: [.ciImage(warmupImage)]
                        )

                        guard await model.generateResult(userInput) != nil else {
                            trace(
                                "warmup returned no valid result image_id=\(warmupEntry.imageID) variant=\(variant.rawValue) output=\(model.output)"
                            )
                            throw BenchmarkError.generationFailed(
                                warmupEntry.imageID,
                                variant,
                                model.output
                            )
                        }

                        trace(
                            "warmup finish index=\(warmupIndex)/\(warmupRuns) image_id=\(warmupEntry.imageID) variant=\(variant.rawValue)"
                        )
                    }
                }
            }

            var records: [BenchmarkRecord] = []
            records.reserveCapacity(entries.count * variants.count)

            for entry in entries {
                trace("entry start image_id=\(entry.imageID)")
                let image = try loadImage(
                    for: entry,
                    relativeTo: manifestURL.deletingLastPathComponent()
                )

                for variant in variants {
                    trace("generation start image_id=\(entry.imageID) variant=\(variant.rawValue)")
                    model.selectedVariant = variant

                    let userInput = UserInput(
                        prompt: .text(entry.prompt),
                        images: [.ciImage(image)]
                    )

                    guard let result = await model.generateResult(userInput) else {
                        trace(
                            "generation returned no valid result image_id=\(entry.imageID) variant=\(variant.rawValue) output=\(model.output)"
                        )
                        throw BenchmarkError.generationFailed(entry.imageID, variant, model.output)
                    }
                    trace("generation finish image_id=\(entry.imageID) variant=\(variant.rawValue)")

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
                    try writeRecords(records, to: outputURL)
                    trace("checkpoint output rows=\(records.count) last_variant=\(variant.rawValue)")

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
                    fflush(stdout)
                }
            }

            trace("encoding records count=\(records.count)")
            trace("writing output path=\(outputURL.path) rows=\(records.count)")
            try writeRecords(records, to: outputURL)
            trace("output write complete path=\(outputURL.path)")

            print("FASTVLM_BENCHMARK PASS all records=\(records.count)")
            print("FASTVLM_BENCHMARK OUTPUT \(outputURL.path)")
            fflush(stdout)
            Darwin.exit(EXIT_SUCCESS)
        } catch {
            trace("FAIL \(error.localizedDescription)")
            Darwin.exit(EXIT_FAILURE)
        }
    }

    private static func benchmarkVariants(_ environment: [String: String]) throws -> [FastVLMVariant] {
        guard let rawValue = environment["FASTVLM_BENCHMARK_VARIANT"],
              !rawValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return FastVLMVariant.allCases
        }

        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "int8", "8", "8-bit":
            return [.int8]
        case "int4", "4", "4-bit":
            return [.int4]
        default:
            throw BenchmarkError.invalidVariant(rawValue)
        }
    }

    private static func benchmarkWarmupRuns(_ environment: [String: String]) throws -> Int {
        guard let rawValue = environment["FASTVLM_BENCHMARK_WARMUP_RUNS"],
              !rawValue.isEmpty else {
            return 0
        }
        guard let value = Int(rawValue), value >= 0 else {
            throw BenchmarkError.invalidWarmupRuns(rawValue)
        }
        return value
    }

    private static func loadImage(
        for entry: BenchmarkManifestEntry,
        relativeTo manifestDirectory: URL
    ) throws -> CIImage {
        let imageURL = URL(
            fileURLWithPath: entry.imagePath,
            relativeTo: manifestDirectory
        ).standardizedFileURL
        let imageData = try Data(contentsOf: imageURL)
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw BenchmarkError.invalidImage(entry.imageID, imageURL.path)
        }
        let image = CIImage(cgImage: cgImage)
        trace("image decoded image_id=\(entry.imageID) frame=0 path=\(imageURL.path)")
        return image
    }

    private static func trace(_ message: String) {
        let line = "FASTVLM_BENCHMARK TRACE \(message)"
        print(line)
        fflush(stdout)

        guard let summaryPath = ProcessInfo.processInfo.environment["GITHUB_STEP_SUMMARY"],
              !summaryPath.isEmpty else {
            return
        }

        let summaryURL = URL(fileURLWithPath: summaryPath)
        let data = Data(("- `\(line)`\n").utf8)
        do {
            if !FileManager.default.fileExists(atPath: summaryURL.path) {
                FileManager.default.createFile(atPath: summaryURL.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: summaryURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
        } catch {
            print("FASTVLM_BENCHMARK TRACE summary-write-failed: \(error.localizedDescription)")
            fflush(stdout)
        }
    }

    private static func writeRecords(_ records: [BenchmarkRecord], to outputURL: URL) throws {
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
        case invalidVariant(String)
        case invalidWarmupRuns(String)
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
                return "Benchmark manifest is empty."
            case .invalidVariant(let value):
                return "FASTVLM_BENCHMARK_VARIANT must be int8 or int4, found '\(value)'."
            case .invalidWarmupRuns(let value):
                return "FASTVLM_BENCHMARK_WARMUP_RUNS must be a non-negative integer, found '\(value)'."
            case .invalidImage(let imageID, let path):
                return "Could not decode benchmark image \(imageID) at \(path)."
            case .generationFailed(let imageID, let variant, let output):
                return "Benchmark generation failed for \(imageID) / \(variant.rawValue): \(output)"
            case .encodingFailed:
                return "Could not encode benchmark JSONL output."
            }
        }
    }
}
#endif
