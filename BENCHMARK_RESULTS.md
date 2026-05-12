# Benchmark Results — gemma4-turbo vs gemma4 base
# Tool: benchmark.py (Ollama API, num_predict=200)

---

## Hardware: Intel Xeon E-2236, 6C/12T — CPU-only (Date: 2026-05-09)

### E2B
| Prompt     | Base tok/s | Turbo tok/s | Speedup |
|------------|-----------|------------|---------|
| short      |      16.6 |       18.0 |   1.08x |
| reasoning  |      16.1 |       17.7 |   1.10x |
| code       |      16.9 |       18.0 |   1.06x |
| **Average**|           |            | **1.08x** |

Size: base 7.2 GB → turbo 4.3 GB (-40%)

### E4B
| Prompt     | Base tok/s | Turbo tok/s | Speedup |
|------------|-----------|------------|---------|
| short      |       9.1 |       10.4 |   1.14x |
| reasoning  |       9.0 |        9.5 |   1.06x |
| code       |       9.0 |        9.4 |   1.04x |
| **Average**|           |            | **1.08x** |

Size: base 9.6 GB → turbo 6.1 GB (-36%)

---

## Hardware: Intel Core i7 + NVIDIA RTX 2070 Max-Q (8 GB VRAM) — GPU inference (Date: 2026-05-11)

### E2B
| Prompt     | Base tok/s | Turbo tok/s | Speedup |
|------------|-----------|------------|---------|
| short      |      16.1 |       18.1 |   1.12x |
| reasoning  |      17.1 |       17.9 |   1.05x |
| code       |      16.0 |       17.9 |   1.12x |
| **Average**|           |            | **1.10x** |

VRAM usage: turbo e2b ~3.8 GB (fits in 4 GB iGPU cards)

### E4B
| Prompt     | Base tok/s | Turbo tok/s | Speedup |
|------------|-----------|------------|---------|
| short      |       9.1 |       11.0 |   1.20x |
| reasoning  |       9.0 |        9.5 |   1.05x |
| code       |       9.1 |        9.5 |   1.05x |
| **Average**|           |            | **1.10x** |

VRAM usage: turbo e4b ~4.5 GB (leaves 3.5 GB free on 8 GB cards)

---

## Flash Attention (e4b, OLLAMA_FLASH_ATTENTION=1)
Short/medium prompts: ~17-18 tok/s (~2x vs baseline)
Source: README benchmarks on same hardware

## Summary
Both sizes show consistent **~8-14% faster eval AND smaller file sizes** across
CPU-only and GPU hardware. IQ4_XS is applied to BF16 source weights (not
re-quantizing already-quantized weights), preserving more information per bit.

| Size | Base size | Turbo size | Reduction | Avg speedup |
|------|-----------|------------|-----------|-------------|
| e2b  | 7.2 GB    | 4.3 GB     | -40%      | 1.10x       |
| e4b  | 9.6 GB    | 6.1 GB     | -36%      | 1.10x       |
| 26b  | 17 GB     | 15 GB      | -12%      | —           |
| 31b  | 19 GB     | 18 GB      | -5%       | —           |
