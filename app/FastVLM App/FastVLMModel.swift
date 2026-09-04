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
            return "int8"
        case .int4:
            return "int4"
        }
    }
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

                let result = try await modelContainer.perform { context in
                    // Measure the time it takes to prepare the input

                    Task { @MainActor in
                        evaluationState = .processingPrompt
                    }

                    let llmStart = Date()
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
                            let llmDuration = Date().timeIntervalSince(llmStart)
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                evaluationState = .generatingResponse
                                self.output = text
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

                        if tokens.count >= maxTokens {
                            return .stop
                        } else {
                            return .more
                        }
                    }

                    // Return the duration of the LLM and the result
                    return result
                }

                // Check if task was cancelled before updating UI
                if !Task.isCancelled {
                    self.output = result.output
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

    private func resetForVariantChange() {
        currentTask?.cancel()
        currentTask = nil
        loadState = .idle
        running = false
        evaluationState = .idle
        modelInfo = ""
        output = ""
        promptTime = ""
    }

    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
        running = false
        output = ""
        promptTime = ""
    }
}
