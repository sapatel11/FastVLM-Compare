//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import CoreImage
import FastVLM
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import MLXVLM

enum FastVLMVariant: String, CaseIterable, Identifiable, Sendable {
    case int8 = "8-bit"
    case int4 = "4-bit"

    var id: Self { self }

    var directoryName: String {
        switch self {
        case .int8:
            return "int8.bundle"
        case .int4:
            return "int4.bundle"
        }
    }
}

struct FastVLMRunResult: Identifiable, Sendable {
    let variant: FastVLMVariant
    let caption: String
    let timeToFirstToken: TimeInterval
    let totalLatency: TimeInterval
    let generatedTokenCount: Int
    let tokensPerSecond: Double

    var id: FastVLMVariant { variant }
}

private enum FastVLMVariantError: LocalizedError {
    case missingBundleResources
    case missingModelDirectory(FastVLMVariant)

    var errorDescription: String? {
        switch self {
        case .missingBundleResources:
            return "FastVLM bundle resources are unavailable."
        case .missingModelDirectory(let variant):
            return "Could not find the \(variant.rawValue) model resources."
        }
    }
}

@Observable
@MainActor
class FastVLMModel {

    public var running = false
    public var modelInfo = ""
    public var output = ""
    public var promptTime: String = ""
    public var timeToFirstToken: TimeInterval = 0
    public var totalLatency: TimeInterval = 0
    public var generatedTokenCount = 0
    public var tokensPerSecond: Double = 0

    public var selectedVariant: FastVLMVariant = .int8 {
        didSet {
            guard selectedVariant != oldValue else { return }
            resetForVariantChange()
        }
    }

    enum LoadState {
        case idle
        case loaded(FastVLMVariant, ModelContainer)
    }

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: 0.0)
    let maxTokens = 240

    var effectiveGenerationTokenLimit: Int {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FASTVLM_BENCHMARK"] == "1" else {
            return maxTokens
        }
        if let rawValue = environment["FASTVLM_BENCHMARK_MAX_TOKENS"],
           let value = Int(rawValue),
           value > 0 {
            return value
        }
        if environment["GITHUB_ACTIONS"] == "true" {
            return 8
        }
        return maxTokens
    }

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 4

    private var loadState = LoadState.idle
    private var currentTask: Task<Void, Never>?

    enum EvaluationState: String, CaseIterable {
        case idle = "Idle"
        case processingPrompt = "Processing Prompt"
        case generatingResponse = "Generating Response"
    }

    public var evaluationState = EvaluationState.idle

    public init() {
        FastVLM.register(modelFactory: VLMModelFactory.shared)
    }

    private func modelConfiguration(for variant: FastVLMVariant) throws -> ModelConfiguration {
        let bundle = Bundle(for: FastVLM.self)
        guard let resourceURL = bundle.resourceURL else {
            throw FastVLMVariantError.missingBundleResources
        }

        let candidateDirectories = [
            resourceURL
                .appendingPathComponent("model", isDirectory: true)
                .appendingPathComponent(variant.directoryName, isDirectory: true),
            resourceURL.appendingPathComponent(variant.directoryName, isDirectory: true),
        ]

        for directory in candidateDirectories {
            let configURL = directory.appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: configURL.path) {
                return ModelConfiguration(directory: directory)
            }
        }

        throw FastVLMVariantError.missingModelDirectory(variant)
    }

    private func _load() async throws -> ModelContainer {
        let variant = selectedVariant

        if case .loaded(let loadedVariant, let modelContainer) = loadState,
           loadedVariant == variant {
            return modelContainer
        }

        // limit the buffer cache
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

        let modelConfiguration = try modelConfiguration(for: variant)
        let modelContainer = try await VLMModelFactory.shared.loadContainer(
            configuration: modelConfiguration
        ) {
            [modelConfiguration, variant] progress in
            Task { @MainActor in
                self.modelInfo =
                    "Loading \(variant.rawValue): \(Int(progress.fractionCompleted * 100))%"
            }
        }

        if selectedVariant == variant {
            self.modelInfo = "Loaded \(variant.rawValue)"
            loadState = .loaded(variant, modelContainer)
        }

        return modelContainer
    }

    public func load() async {
        do {
            _ = try await _load()
        } catch {
            self.modelInfo = "Error loading \(selectedVariant.rawValue) model: \(error.localizedDescription)"
        }
    }

    public func generate(_ userInput: UserInput) async -> Task<Void, Never> {
        if let currentTask, running {
            return currentTask
        }

        resetMetrics()
        running = true

        // Cancel any existing task
        currentTask?.cancel()

        // Create new task and store reference
        let task = Task {
            do {
                let modelContainer = try await _load()

                // each time you generate you will get something new
                MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

                // Check if task was cancelled
                if Task.isCancelled { return }

                let (result, measuredTotalLatency) = try await modelContainer.perform { context in
                    Task { @MainActor in
                        evaluationState = .processingPrompt
                    }

                    // Benchmark timing begins after model loading and before input preparation.
                    let inferenceStart = Date()
                    let input = try await context.processor.prepare(input: userInput)

                    var seenFirstToken = false

                    // FastVLM generates the output
                    let result = try MLXLMCommon.generate(
                        input: input, parameters: generateParameters, context: context
                    ) { tokens in
                        // Check if task was cancelled
                        if Task.isCancelled {
                            return .stop
                        }

                        if !seenFirstToken {
                            seenFirstToken = true

                            // produced first token, update the time to first token,
                            // the processing state and start displaying the text
                            let llmDuration = Date().timeIntervalSince(inferenceStart)
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                evaluationState = .generatingResponse
                                self.output = text
                                self.timeToFirstToken = llmDuration
                                self.promptTime = "\(Int(llmDuration * 1000)) ms"
                            }
                        }

                        // Show the text in the view as it generates
                        if tokens.count % displayEveryNTokens == 0 {
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                self.output = text
                            }
                        }

                        if tokens.count >= effectiveGenerationTokenLimit {
                            return .stop
                        } else {
                            return .more
                        }
                    }

                    let measuredTotalLatency = Date().timeIntervalSince(inferenceStart)
                    return (result, measuredTotalLatency)
                }

                // Check if task was cancelled before updating UI
                if !Task.isCancelled {
                    self.output = result.output
                    self.totalLatency = measuredTotalLatency
                    self.generatedTokenCount = result.tokens.count
                    self.tokensPerSecond = result.tokensPerSecond
                }

            } catch {
                if !Task.isCancelled {
                    output = "Failed: \(error)"
                }
            }

            if evaluationState == .generatingResponse {
                evaluationState = .idle
            }

            running = false
        }

        currentTask = task
        return task
    }

    func generateResult(_ userInput: UserInput) async -> FastVLMRunResult? {
        let variant = selectedVariant
        let task = await generate(userInput)
        _ = await task.result

        let caption = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard selectedVariant == variant,
              !caption.isEmpty,
              !caption.hasPrefix("Failed:"),
              timeToFirstToken > 0,
              totalLatency >= timeToFirstToken,
              generatedTokenCount > 0,
              tokensPerSecond.isFinite,
              tokensPerSecond > 0 else {
            return nil
        }

        return FastVLMRunResult(
            variant: variant,
            caption: caption,
            timeToFirstToken: timeToFirstToken,
            totalLatency: totalLatency,
            generatedTokenCount: generatedTokenCount,
            tokensPerSecond: tokensPerSecond
        )
    }

    private func resetMetrics() {
        promptTime = ""
        timeToFirstToken = 0
        totalLatency = 0
        generatedTokenCount = 0
        tokensPerSecond = 0
    }

    private func resetForVariantChange() {
        currentTask?.cancel()
        currentTask = nil
        loadState = .idle
        running = false
        evaluationState = .idle
        modelInfo = ""
        output = ""
        resetMetrics()
    }

    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
        running = false
        output = ""
        resetMetrics()
    }
}
