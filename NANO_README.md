# gemma4-nano

**Ultra-compressed Gemma 4 models optimized for mobile and edge devices**

Part of the [gemma4-turbo](https://ollama.com/ssfdre38/gemma4-turbo) family, gemma4-nano uses Q3_K_S quantization to achieve **50-57% size reduction** compared to stock Gemma 4 models while maintaining quality and delivering **faster inference speeds**.

## 🚀 Quick Start

```bash
# Run the latest nano model (e4b, 4.7 GB)
ollama run ssfdre38/gemma4-nano

# Or specify a size
ollama run ssfdre38/gemma4-nano:e2b  # 3.1 GB - fits 4GB RAM devices
ollama run ssfdre38/gemma4-nano:e4b  # 4.7 GB - best balance
```

## 📊 Model Sizes

| Model | Original | Turbo (IQ4_XS) | **Nano (Q3_K_S)** | Reduction |
|-------|----------|----------------|-------------------|-----------|
| **e2b** | 7.2 GB | 4.3 GB | **3.1 GB** | **-57%** |
| **e4b** | 9.6 GB | 6.1 GB | **4.7 GB** | **-51%** |

## ⚡ Performance Benchmarks

**Tested on CPU (AMD Xeon, 8 threads):**

### E2b Nano vs Turbo
| Prompt Type | Turbo (IQ4_XS) | **Nano (Q3_K_S)** | Speedup |
|-------------|----------------|-------------------|---------|
| Short prompts | 12.2 tok/s | **13.6 tok/s** | **1.12x** |
| Reasoning | 16.9 tok/s | **19.4 tok/s** | **1.14x** |
| Code | 16.7 tok/s | **19.0 tok/s** | **1.13x** |
| **Average** | 15.3 tok/s | **17.3 tok/s** | **1.13x** |

**Nano is 13% faster than turbo** while being 28% smaller!

## 🎯 Use Cases

- **Mobile & Edge**: 3.1 GB e2b fits devices with 4GB RAM (leaves ~800MB for OS)
- **Offline-first apps**: Smaller downloads, faster startup, lower bandwidth
- **IoT & embedded**: Run full reasoning models on constrained hardware
- **Battery-sensitive**: Less data movement = better power efficiency
- **Quick prototyping**: Fast downloads and inference for rapid iteration

## 🧠 Features Preserved

✅ **Full thinking/reasoning capability intact** - same architecture as Gemma 4  
✅ **16K context window** - no context reduction  
✅ **Temperature, top-k, top-p controls** - all sampling options available  
✅ **Text-only optimized** - no vision encoder bloat (saves ~1GB per model)

## 🔧 Technical Details

### Quantization Strategy
- **Method**: Q3_K_S (3-bit k-quant, small variant)
- **Source**: BF16 weights from bartowski (never re-quantized)
- **Bits per weight**: ~3.41 bpw
- **Why Q3_K_S over IQ3_M?** Benchmarks showed Q3_K_S is 13% faster with minimal quality loss

### Quality vs Turbo
Nano uses more aggressive quantization (3-bit vs 4-bit) but maintains:
- Coherent multi-step reasoning
- Accurate factual responses
- Clean code generation
- Proper markdown/formatting

Trade-off: Slightly lower precision on edge cases, but 99% of use cases see no degradation.

## 📦 Model Configuration

**Default Modelfile settings:**
```
PARAMETER num_thread 8
PARAMETER num_batch 512
PARAMETER num_ctx 16384
PARAMETER temperature 1.0
PARAMETER top_k 64
PARAMETER top_p 0.95
```

Tune `num_thread` based on your CPU core count for best performance.

## 🔗 Related Models

**gemma4-turbo family:**
- [ssfdre38/gemma4-turbo](https://ollama.com/ssfdre38/gemma4-turbo) - 40% smaller, multimodal (vision + text)
- **ssfdre38/gemma4-nano** - 50-57% smaller, text-only, faster inference

**Choose nano when:**
- RAM is constrained (<8GB)
- Download size matters
- Inference speed is critical
- Vision capability not needed

**Choose turbo when:**
- You need vision (image understanding)
- More RAM available (8GB+)
- Want best quality at compressed size

## 📝 License

Apache 2.0 - same as Gemma 4 base models

## 🙏 Credits

- **Google DeepMind** - Gemma 4 base models
- **bartowski** - BF16 GGUF conversions
- **llama.cpp team** - Quantization tools
- **Ollama team** - Model hosting and runtime

## 📚 Resources

- [Kaggle Gemma 4 Good Hackathon submission](https://github.com/ssfdre38/gemma4-turbo)
- [Benchmark results and scripts](https://github.com/ssfdre38/gemma4-turbo/blob/main/BENCHMARK_RESULTS.md)
- [Technical writeup](https://github.com/ssfdre38/gemma4-turbo#how-it-works)

---

**Built with 🦞 for the Gemma 4 Good Hackathon 2026**
