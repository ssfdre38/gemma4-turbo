---
license: apache-2.0
base_model:
  - google/gemma-4-E2B-it
  - google/gemma-4-E4B-it
  - google/gemma-4-26B-A4B-it
  - google/gemma-4-31B-it
tags:
  - gemma4
  - gemma4.5
  - gguf
  - iq4_xs
  - quarot
  - hadamard
  - multimodal
  - vision
  - cpu-inference
  - ollama
  - llama-cpp
  - haven-cpp
---

# Gemma 4.5 Turbo (`ssfdre38/gemma4.5-turbo`)

**Algebraically optimized Gemma 4 models built from unquantized BF16 source weights via [`haven-bake`](https://github.com/Haven-AI-Companion/haven-cpp) — lower perplexity, smaller footprint, faster CPU inference, and full multimodal vision + thinking preserved.**

> **Zero retraining. Pure exact linear algebra baked directly into BF16 weights before `IQ4_XS` quantization.**

## Quick Start (Ollama — Multimodal Vision + Thinking)

All Ollama tags use the universal stock-compatible `--mode stock` Gemma 4.5 weights merged with the full multimodal vision encoder (`mmproj`), so they run out of the box on stock **Ollama**, **`llama.cpp`**, and **`haven-cpp`**:

```bash
ollama run ssfdre38/gemma4.5-turbo          # e4b (5.63 GiB, recommended, multimodal)
ollama run ssfdre38/gemma4.5-turbo:e2b      # 3.99 GiB, 8 GB+ RAM (multimodal)
ollama run ssfdre38/gemma4.5-turbo:e4b      # 5.63 GiB, 10 GB+ RAM (multimodal)
ollama run ssfdre38/gemma4.5-turbo:26b      # 14.21 GiB, 26B-A4B MoE (multimodal)
ollama run ssfdre38/gemma4.5-turbo:31b      # 16.82 GiB, 31B Dense flagship (multimodal)
```

Vision works out of the box:
```bash
ollama run ssfdre38/gemma4.5-turbo "describe this image" /path/to/image.jpg
```

---

## What Changed in Gemma 4.5 Turbo?

Gemma 4 introduces several architectural quirks that waste memory or create severe 4-bit quantization cliffs when quantized naively:
- **Dead placeholder shared-KV tensors:** E2B (`shared_kv_layers=20`) and E4B (`shared_kv_layers=18`) ship 60 and 54 unused `attn_k`, `attn_v`, and `attn_k_norm` tensors on shared-KV layers.
- **Unfolded MoE router & expert scales:** 26B-A4B ships separate `ffn_gate_inp.scale` and `ffn_down_exps.scale` multipliers.
- **Attention & activation outlier cliffs:** Learned RMSNorm gammas reach `450x–900x`, causing severe clipping in 4-bit `IQ4_XS` weight blocks.

Our open-source C++20 AVX2 optimizer **`haven-bake`** resolves these at BF16 bake time *prior* to `llama-quantize IQ4_XS`:

1. **Universal Stock Edition (`gemma4.5-*-iq4xs.gguf` & `*-mmproj.gguf` — 100% Ollama & `llama.cpp` compatible):**
   - Strips dead placeholder shared-KV tensors (`60` tensors / `37.8 MB` BF16 on E2B; `54` tensors / `33.6 MB` IQ4_XS on E4B).
   - Folds MoE `ffn_gate_inp.scale` and `ffn_down_exps.scale` cleanly into weights (`26B-A4B`).
   - Bakes **zero-cost QuaRot R2 head-dimension Hadamard rotation (`V → O`)** on `E4B` (`24/24` KV-source layers), `26B-A4B` (`25/30` SWA layers, automatically skipping `5` `k_eq_v` layers), and `31B` (`50/60` SWA layers, automatically skipping `10` `k_eq_v` layers). Because Gemma 4's V-RMSNorm is parameter-free and `H` is orthogonal, `V → O` cancels identically at runtime with **0 extra operations** while smoothing 4-bit quantization noise. *(On E2B's single-KV-head geometry, `--no-r2` proved strictly optimal for stock perplexity.)*
   - Merges the full multimodal vision encoder (`merge_g45_mmproj.py`) without re-quantizing or disturbing LM weights.

2. **Native `haven-cpp` Turbo Edition (`gemma4.5-*-haven-turbo-iq4xs.gguf`):**
   - Everything in Stock, PLUS **norm-preserving orthogonal Block-256 / Block-64 Fast Walsh-Hadamard Transform (FWHT)** baked into `Q/K/V`, `FFN Gate/Up`, `FFN Down`, and `MoE` projections (`gemma4.turbo.*` metadata auto-dispatched by [`haven-cpp`](https://github.com/Haven-AI-Companion/haven-cpp) AVX2 kernels).
   - Flattens both weight-block crest factors and `450x–900x` runtime activation cliffs by `16x`, slashing KL divergence vs unquantized BF16 by **19% to 44%**!

---

## Verified Benchmarks (CPU-Only: Intel Xeon E-2236, 6C/12T AVX2, No GPU)

### 1. Stock `llama.cpp` Perplexity (`llama-perplexity`, `calibration_data.txt`, `n_ctx=512`, `t=12`)

Every single model size achieves **equal or strictly lower perplexity** than original `gemma4-turbo` with **zero runtime overhead**:

| Model | Original `gemma4-turbo` PPL | **Gemma 4.5 Stock PPL** | Improvement | Text-Only GGUF | Multimodal (`mmproj`) GGUF | vs Stock Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E2B** | `8.6825` | **`8.6825`** *(0.0000 loss, 60 dead KV stripped)* | Faster (`25.5s` vs `26.8s`), `-11 MiB` RAM | `3.07 GiB` | **`3.99 GiB`** | `-45%` *(was 7.2 GB)* |
| **E4B** | `6.2341` | **`5.9685`** *(54 dead KV stripped + 24/24 R2)* | **`-0.266` PPL better** (all 3 chunks) | `4.71 GiB` | **`5.63 GiB`** | `-41%` *(was 9.6 GB)* |
| **26B-A4B MoE** | `33.9742` | **`33.0918`** *(folded scales + 25/30 SWA R2)* | **`-0.882` PPL better** | `13.10 GiB` | **`14.21 GiB`** | `-16%` *(was 17 GB)* |
| **31B Dense** | `11.7080` | **`10.9936`** *(50/60 SWA R2)* | **`-0.714` PPL better** (both chunks) | `15.70 GiB` | **`16.82 GiB`** | `-11%` *(was 19 GB)* |

### 2. Live Ollama CPU Throughput (`POST /api/chat`, Intel Xeon E-2236, 6C/12T, No GPU)

Verified end-to-end on stock Ollama (`think: false`, `num_ctx: 2048`, exact clean stop):

| Ollama Tag | Size | Prefill (`tok/s`) | Decode (`tok/s`) | Status |
| :--- | :---: | :---: | :---: | :---: |
| `ssfdre38/gemma4.5-turbo:e2b` | `4.3 GB` | **`116.91 tok/s`** | **`18.89 tok/s`** | Verified |
| `ssfdre38/gemma4.5-turbo:e4b` (`:latest`) | `6.0 GB` | **`61.65 tok/s`** | **`9.93 tok/s`** | Verified |
| `ssfdre38/gemma4.5-turbo:26b` | `15 GB` | **`55.29 tok/s`** | **`11.63 tok/s`** | Verified |
| `ssfdre38/gemma4.5-turbo:31b` | `18 GB` | **`8.81 tok/s`** | **`1.80 tok/s`** | Verified |

### 3. Native `haven-cpp` Turbo vs Unquantized BF16 Ground Truth

Measured against the original unquantized BF16 masters on CPU (`haven-cpp` AVX2 L2-tiled integer GEMM + fused FWHT):

| Model (`haven-turbo`) | Metric vs Unquantized BF16 | Original `gemma4-turbo` | **Gemma 4.5 Native Turbo** | Gain | Prefill (`S=274`, 12T) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **E2B Turbo** | `KL(bf16 \|\| q)` | `1.4460` | **`0.8139`** | **`-43.7%` less KL** | **`108.7 tok/s`** *(Stock: `113.2 tok/s`, TTFT `2.42s`)* |
| **E2B Turbo** | `forward_batch` Cosine vs BF16 | `0.8536` | **`0.9818`** | **`8.0x` less error** | Decode: `16.1 tok/s` |
| **E4B Turbo** | `KL(bf16 \|\| q)` | `0.8067` | **`0.6523`** | **`-19.1%` less KL** | **`58.64 tok/s`** *(TTFT `4.67s`)* |
| **E4B Turbo** | `forward` Cosine / Top-1 Agree | `0.9904` / `62.5%` | **`0.9983` / `72.9%`** | **`5.6x` less error / `+10.4%` Top-1** | Decode: `9.37 tok/s` |
| **26B-A4B Turbo** | `seq` vs `batch` Cosine / Top-1 | — | **`0.9998` / Exact `100==100`** | Clean English + FWHT | **`58.70 tok/s`** *(TTFT `4.67s`, `5.07x` speedup)* |
| **31B Dense Turbo** | `seq` vs `batch` Cosine / Top-1 | `0.9991` *(Stock)* | **`0.9998` / Exact `100==100`** | `max_abs_diff` cut `1.46 → 0.83` (`-43%`) | **`7.57 tok/s`** *(Decode: `1.69 tok/s` vs `1.62` Stock)* |

---

## Included Files on Hugging Face (`ssfdre38/gemma4.5-turbo-gguf`)

### Universal GGUFs (Ollama / `llama.cpp` / LM Studio / `haven-cpp`)
- **Multimodal (Vision + Text + Thinking):**
  - `gemma4.5-e2b-iq4xs-mmproj.gguf` (`3.99 GiB`)
  - `gemma4.5-e4b-iq4xs-mmproj.gguf` (`5.63 GiB`)
  - `gemma4.5-26b-iq4xs-mmproj.gguf` (`14.21 GiB`)
  - `gemma4.5-31b-iq4xs-mmproj.gguf` (`16.82 GiB`)
- **Text-Only (Ultra-lean, no vision tower):**
  - `gemma4.5-e2b-iq4xs.gguf` (`3.07 GiB`)
  - `gemma4.5-e4b-iq4xs.gguf` (`4.71 GiB`)
  - `gemma4.5-26b-iq4xs.gguf` (`13.10 GiB`)
  - `gemma4.5-31b-iq4xs.gguf` (`15.70 GiB`)

### Native `haven-cpp` Block-FWHT Turbo GGUFs
> Requires [`haven-cpp`](https://github.com/Haven-AI-Companion/haven-cpp) (commit `637ffa6` or later) which automatically detects `gemma4.turbo.*` metadata and runs fused AVX2 FWHT kernels. Do **not** load `*-haven-turbo-*` files in stock `llama.cpp`/Ollama (use the universal files above instead).

- `gemma4.5-e2b-haven-turbo-iq4xs.gguf` (`3.07 GiB`)
- `gemma4.5-e4b-haven-turbo-iq4xs.gguf` (`4.71 GiB`)
- `gemma4.5-26b-haven-turbo-iq4xs.gguf` (`13.10 GiB`)
- `gemma4.5-31b-haven-turbo-iq4xs.gguf` (`15.70 GiB`)

---

## Reproducibility

Built with open-source tooling:
- **Architecture Optimizer:** `haven-bake` ([Haven-AI-Companion/haven-cpp](https://github.com/Haven-AI-Companion/haven-cpp))
- **Quantizer:** `llama.cpp` b9050 `llama-quantize` (`IQ4_XS`)
- **Multimodal Merger:** `merge_g45_mmproj.py` ([ssfdre38/gemma4-turbo](https://github.com/ssfdre38/gemma4-turbo))

## License

Derived from Google's Gemma 4 IT model family (`google/gemma-4-{E2B,E4B,26B-A4B,31B}-it`) under the **Apache 2.0 License** and [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
Optimization, baking, and benchmarking by [ssfdre38](https://ollama.com/ssfdre38).
