# FastVLM-Compare benchmark

This directory contains the controlled measurement workflow for the FastVLM-0.5B Stage 3 INT8-vs-INT4 comparison.

The project's final measurements are now produced on a **GitHub-hosted Apple Silicon macOS runner** because no dedicated local Mac is available. The original one-image workflow remains an infrastructure smoke test; final results come only from the separate manually triggered workflow:

`FastVLM final hosted benchmark`

This is an explicit experimental limitation. The final report must describe the environment as GitHub-hosted Apple Silicon and must not claim that the results are dedicated bare-metal or local-Mac measurements.

## Final protocol

The final hosted protocol is intentionally small but repeatable:

- Environment: GitHub-hosted `macos-15` Apple Silicon runner.
- Model: the same FastVLM-0.5B Stage 3 checkpoint for both variants.
- Build: Release configuration.
- Vision encoder: shared between INT8 and INT4.
- LLM quantization: 8-bit vs 4-bit, group size 64.
- Decoding: temperature `0.0`.
- Generation cap: `64` tokens.
- Dataset: five fixed repository assets in `benchmark_manifest.json`.
- Warm-up: one generation using the first manifest entry in each fresh variant process.
- Measured repetitions: five.
- Process isolation: INT8 and INT4 run in separate application processes.
- Ordering: odd repetitions run INT8 then INT4; even repetitions run INT4 then INT8.
- Model loading: excluded from TTFT and total-latency metrics.
- TTFT start: immediately before multimodal input preparation.
- TTFT end: first generated token.
- Total-latency start: same point as TTFT.
- Total-latency end: generation completion.

With five images and five repetitions, the workflow produces **50 measured records**: 25 INT8 and 25 INT4. Warm-up generations are not written to the measurement JSONL.

### Why separate processes?

The existing CI smoke harness can execute both variants in one process because its job is only to prove that the native benchmark works.

The final benchmark instead starts a fresh application process for each variant session. This reduces cross-variant contamination from model residency, MLX state, and caches. Alternating the variant order between repetitions also reduces systematic order effects.

## Run the final benchmark

The final benchmark is manual-only. It will never start automatically.

1. Open the repository on GitHub.
2. Open **Actions**.
3. Select **FastVLM final hosted benchmark**.
4. Choose **Run workflow**.
5. Run it on `main`.

The workflow performs the complete experiment in one hosted job:

```text
checkout
  ↓
prepare pinned FastVLM-0.5B Stage 3 checkpoint
  ↓
export shared Core ML vision encoder
  ↓
convert 8-bit LLM
  ↓
convert 4-bit LLM
  ↓
build Release macOS app
  ↓
1 warm-up + 5 measured repetitions
  ↓
alternate INT8/INT4 order in fresh processes
  ↓
50 raw measurements
  ↓
aggregate analysis
  ↓
upload results artifact
```

The workflow does **not** upload model checkpoints, quantized model bundles, or the built application.

## Results artifact

A successful or partially completed run uploads:

`fastvlm-final-benchmark-results`

The artifact contains the benchmark data and diagnostics only:

```text
metadata.json
measurements.jsonl
raw/
logs/
analysis-console.txt
analysis/
    summary.json
    summary.md
    per_image.csv
    captions.md
```

`metadata.json` records the runner/macOS/hardware information, repository commit, manifest hash, protocol settings, and discovered model-bundle sizes.

`measurements.jsonl` contains the measured rows plus:

- `repetition`
- `order_in_repetition`

The analysis reports:

- median and mean TTFT
- median and mean total latency
- median and mean tokens/second
- median generated-token count
- INT4 relative improvement for TTFT, total latency, and throughput
- INT4 LLM-bundle storage reduction
- effective LLM-plus-shared-vision storage reduction
- exact caption agreement as a reproducibility signal

`captions.md` presents paired captions for manual quality review. Exact string agreement is **not** treated as a semantic quality metric.

## Dataset

`benchmark_manifest.json` is a small fixed suite using assets already versioned in the repository. It covers:

- chart/diagram understanding
- general visual description
- counting
- emoji/symbol interpretation
- handwriting/text reading

Animated GIF assets are decoded deterministically from frame 0 by the Swift benchmark runner.

This is not intended to replace a large academic multimodal benchmark. Its purpose is to provide a fixed, reproducible set of distinct visual tasks for the engineering comparison.

The existing `benchmark_manifest.example.json` remains the one-image smoke-test manifest.

## Interpreting the results

A lower TTFT is better.

A lower total latency is generally better, but total latency is also affected by how many tokens each variant generates. Read it together with generated-token count and tokens/second.

A higher tokens/second value is better for decoder throughput.

Model loading is intentionally excluded from TTFT and total latency. This keeps the experiment focused on multimodal input preparation plus inference/generation.

For output quality, inspect whether each caption preserves the correct objects, text, counts, and important details; remains coherent; and avoids new hallucinations. Do not infer semantic quality from exact-string matching alone.

Most importantly, the final conclusion must remain scoped to the environment actually measured:

> These results compare FastVLM-0.5B INT8 and INT4 under the same controlled procedure on a GitHub-hosted Apple Silicon macOS runner. Hosted-runner virtualization and shared-infrastructure variability are limitations, so the results should not be presented as dedicated bare-metal Mac performance.

## Optional local path

The local helpers remain available if a physical Apple Silicon Mac becomes available later:

```bash
PYTHON_BIN=python3.10 bash benchmark/prepare_local_app.sh
python3 benchmark/run_local_benchmark.py --app "/absolute/path/to/FastVLM App.app"
```

That local path uses the same five-image, five-repetition protocol and can provide a future bare-metal comparison without redesigning the benchmark.
