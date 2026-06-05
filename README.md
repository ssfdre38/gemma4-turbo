# Gemma 4 Turbo

A fully optimized Gemma 4 family built from bf16 source weights — smaller, faster, and fully multimodal out of the box.

> 🚀 **New:** [gemma4-nano](https://ollama.com/ssfdre38/gemma4-nano) — ultra-compressed text-only models (3-5 GB) for mobile and edge devices, 13% faster than turbo!

## Why Turbo?

Google's stock Gemma 4 on Ollama ships with Q4_K_M quantization applied to already-quantized weights. Gemma 4 Turbo starts from the original bf16 source and applies **IQ4_XS** (4.25 bpw non-linear quantization), producing better quality at a smaller size. Full vision + thinking capabilities are preserved.

| Tag | Size | vs Base | RAM Required |
|-----|------|---------|--------------|
| `e2b` | 4.3 GB | -40% (was 7.2 GB) | 8 GB+ |
| `e4b` / `latest` | 6.1 GB | -36% (was 9.6 GB) | 10 GB+ |
| `12b` | 6.9 GB | -31% (was 10 GB) | 12 GB+ |
| `26b` | 15 GB | -12% (was 17 GB) | 20 GB+ |
| `31b` | 18 GB | -5% (was 19 GB) | 24 GB+ |

## Quick Start

```bash
ollama run ssfdre38/gemma4-turbo          # e4b (recommended, multimodal)
ollama run ssfdre38/gemma4-turbo:e2b      # 8GB RAM machines
ollama run ssfdre38/gemma4-turbo:12b      # balanced multimodal
ollama run ssfdre38/gemma4-turbo:26b      # high quality
ollama run ssfdre38/gemma4-turbo:31b      # maximum quality

# Ultra-compressed text-only (nano family)
ollama run ssfdre38/gemma4-nano           # 4.7 GB, 13% faster, mobile-optimized
ollama run ssfdre38/gemma4-nano:e2b       # 3.1 GB, fits 4GB RAM devices
```

## Vision & Multimodal

All tags include the full vision encoder — text, images, and thinking are all supported:

```bash
ollama run ssfdre38/gemma4-turbo "describe this image" /path/to/image.jpg
```

## Performance

Benchmarked on CPU (Intel Xeon E-2236, 6C/12T, no GPU). All runs clean with no competing processes.

### Size & RAM (e4b)

| | Base gemma4:e4b | Turbo e4b | Savings |
|--|--|--|--|
| Model size | 9.6 GB | 6.1 GB | **-36%** |
| RAM loaded | ~9.6 GB | ~6.5 GB | **-32%** |

### Tokens per Second (e4b, 8 threads)

| Prompt type | Base tok/s | Turbo tok/s | With Flash Attn |
|-------------|-----------|-------------|-----------------|
| Short (1 sentence) | 10.3 | 10.6 | **~17–18** |
| Reasoning (200 tok) | 9.2 | 9.5 | 9.8 |
| Code generation (200 tok) | 9.1 | 9.6 | 9.8 |

> Flash attention delivers the biggest gain on short conversations — the most common real-world use case.

### Prefill Speed (time to first token, 8 threads, no flash attn)

| Prompt length | Prefill time |
|---------------|-------------|
| Short (~10 tokens) | ~2.9s |
| Medium (~80 tokens) | ~6.9s |
| Long (~480 tokens) | ~21.6s |

## Speed Tip — Enable Flash Attention

Set this environment variable before starting Ollama for roughly **2x faster token generation** on short and medium-length conversations:

**Windows:**
```powershell
[System.Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "Machine")
```

**macOS/Linux:**
```bash
echo 'OLLAMA_FLASH_ATTENTION=1' >> ~/.bashrc   # or ~/.zshrc
```

Then restart Ollama. Also recommended:

```bash
OLLAMA_KV_CACHE_TYPE=q8_0    # halves KV cache RAM usage
```

## What Makes This Different From a Repackage

- **IQ4_XS applied to bf16 source** — not re-quantizing already-quantized weights. Source weights downloaded from bartowski's bf16 GGUFs, quantized with llama.cpp `llama-quantize`.
- **Vision encoder preserved** — original Ollama blobs contain 1411 vision tensors (`a.*`, `v.*`, `mm.*`) plus `gemma4.vision.*` KV metadata. These are merged back into the IQ4_XS LM weights so nothing is lost.
- **Tuned defaults** — `num_thread 8`, `num_batch 512`, `num_ctx 16384` benchmarked and optimized for CPU inference. Setting threads to the logical processor count (e.g. 12 on a 6C/12T CPU) kills eval speed via hyperthreading contention — physical core count is the sweet spot.

## Technical Details

- **Quantization:** IQ4_XS (4.25 bpw, non-linear, importance matrix sampling)
- **Source:** bartowski bf16 GGUFs → llama.cpp b9050 `llama-quantize`
- **Vision:** merged from original `gemma4:e4b/e2b/26b/31b` Ollama blobs (2131 tensors total for e2b/e4b; 1076 for 26b/31b)
- **Architecture:** `gemma4` with full multimodal projector

## See It In Action

### Ash Server — Secure AI Backend (Recommended)

[**Ash Server**](https://github.com/ssfdre38/ash-server) is a production-grade, security-first AI backend built on .NET 10, designed as a hardened platform for running `gemma4-turbo` locally with full access control.

**What Ash Server provides:**
- 🔐 JWT auth + full RBAC roles and permissions
- 🛡️ Prompt injection guard (7 patterns) + per-user rate limiting
- 💬 Discord, Slack, and Telegram integrations — permission-gated via identity linking
- 📋 Full audit log for every external chat action
- 🖥️ Admin panel — users, backends, channels, MCP servers, analytics
- 🔌 MCP protocol support (add any MCP tool server)
- 🦙 Runs on commodity hardware — no GPU, 8 GB RAM minimum
- 📦 One-command install: Windows Service, systemd, or launchd

**Quick start:**
```bash
ollama pull ssfdre38/gemma4-turbo
git clone https://github.com/ssfdre38/ash-server
sudo bash ash-server/install.sh    # Linux/macOS — build + install as service
```
```powershell
# Windows (Administrator PowerShell):
git clone https://github.com/ssfdre38/ash-server
cd ash-server; .\install.ps1
```

Open **http://localhost:18799** — first registered user becomes admin automatically.

→ [**github.com/ssfdre38/ash-server**](https://github.com/ssfdre38/ash-server)

---

### Ash Bot — Discord Personality Bot

[**Ash Bot**](https://github.com/ssfdre38/ash-bot) is a Discord bot with Ash's personality, 20 built-in tools, and long-term memory — the original consumer-facing client.

→ [**github.com/ssfdre38/ash-bot**](https://github.com/ssfdre38/ash-bot)

---

## Model Family

**gemma4-turbo** (this repo):
- IQ4_XS quantization (4.25 bpw)
- 35-40% size reduction vs stock models
- **Includes vision encoder** (multimodal)
- 4.3 GB - 18 GB depending on size
- [View on Ollama Hub](https://ollama.com/ssfdre38/gemma4-turbo)

**[gemma4-nano](https://ollama.com/ssfdre38/gemma4-nano)**:
- Q3_K_S quantization (3.41 bpw)
- 30-57% size reduction vs stock models
- **Text-only** (no vision encoder)
- 3.1 GB - 13 GB depending on size
- **13% faster inference** than turbo
- Optimized for mobile and edge devices
- [View documentation](NANO_README.md)

Choose **turbo** when you need vision capability and multimodal support.  
Choose **nano** when RAM is constrained or inference speed is critical.

---

## License

Derived from [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-E4B-it) and family.
Usage governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).  
Quantization and optimization work by [ssfdre38](https://ollama.com/ssfdre38).
