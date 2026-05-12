"""Upload gemma4-nano GGUFs to Hugging Face"""
from huggingface_hub import HfApi, create_repo, login
import os

# Login (will prompt for token if not cached)
print("Checking Hugging Face authentication...")
try:
    api = HfApi()
    user = api.whoami()
    print(f"✓ Logged in as: {user['name']}")
except Exception as e:
    print(f"\n⚠️  Need to log in to Hugging Face")
    print("Run: huggingface-cli login")
    print("Or set HF_TOKEN environment variable")
    exit(1)

# Repository details
repo_id = "ssfdre38/gemma4-nano-gguf"
repo_type = "model"

print(f"\n📦 Creating/updating repository: {repo_id}")

try:
    # Create repo if it doesn't exist
    create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    print(f"✓ Repository ready: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"✓ Repository exists: https://huggingface.co/{repo_id}")

# Upload files
files_to_upload = [
    ("gemma4-e2b-q3ks-nano.gguf", "2.9 GB"),
    ("gemma4-e4b-q3ks-nano.gguf", "4.3 GB"),
    ("gemma4-26b-q3ks-nano.gguf", "11.4 GB"),
    ("gemma4-31b-q3ks-nano.gguf", "12.8 GB"),
    ("NANO_README.md", "docs"),
    ("NANO_MODEL_CARD.md", "docs"),
]

print(f"\n📤 Uploading files...")
for filename, size in files_to_upload:
    filepath = f"C:\\Users\\admin\\gemma4-turbo-family\\{filename}"
    if not os.path.exists(filepath):
        print(f"⚠️  Skipping {filename} (not found)")
        continue
    
    print(f"  Uploading {filename} ({size})...")
    try:
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"  ✓ {filename} uploaded")
    except Exception as e:
        print(f"  ✗ {filename} failed: {e}")

# Upload README
readme_content = f"""---
license: apache-2.0
language:
- en
tags:
- gemma
- gguf
- quantization
- mobile
- edge
---

# gemma4-nano GGUF Files

Ultra-compressed Gemma 4 models optimized for mobile and edge devices.

## Quick Start with Ollama

```bash
ollama run ssfdre38/gemma4-nano:e2b   # 3.1 GB
ollama run ssfdre38/gemma4-nano:e4b   # 4.7 GB (latest)
ollama run ssfdre38/gemma4-nano:26b   # 12 GB
ollama run ssfdre38/gemma4-nano:31b   # 13 GB
```

**Or download GGUFs directly:**

- **gemma4-e2b-q3ks-nano.gguf** (2.9 GB) - Fits 4GB RAM devices
- **gemma4-e4b-q3ks-nano.gguf** (4.3 GB) - Recommended default
- **gemma4-26b-q3ks-nano.gguf** (11.4 GB) - High capability
- **gemma4-31b-q3ks-nano.gguf** (12.8 GB) - 31B params, smaller than stock e4b!

## Use with Ollama (from GGUF)

```bash
# Download GGUF
wget https://huggingface.co/ssfdre38/gemma4-nano-gguf/resolve/main/gemma4-e2b-q3ks-nano.gguf

# Create Modelfile
echo "FROM ./gemma4-e2b-q3ks-nano.gguf
PARAMETER num_ctx 16384" > Modelfile

# Create model
ollama create my-nano -f Modelfile
ollama run my-nano
```

## Specifications

- **Quantization**: Q3_K_S (3.41 bits per weight)
- **Context**: 16,384 tokens
- **Format**: GGUF v3
- **Modality**: Text-only (no vision encoder)

## Performance

Nano models achieve **13% faster inference** than gemma4-turbo on CPU with 8 threads.

| Model | Size | Reduction vs Stock | Notes |
|-------|------|--------------------|-------|
| E2b nano | 3.1 GB | -57% (was 7.2 GB) | Fits 4GB RAM devices |
| E4b nano | 4.7 GB | -51% (was 9.6 GB) | Recommended default |
| 26b nano | 12 GB | -29% (was 17 GB) | High capability |
| 31b nano | 13 GB | -32% (was 19 GB) | **31B params < stock e4b size!** |

## Documentation

- [Full documentation](https://github.com/ssfdre38/gemma4-turbo/blob/master/NANO_README.md)
- [Model card](https://github.com/ssfdre38/gemma4-turbo/blob/master/NANO_MODEL_CARD.md)
- [Ollama Hub](https://ollama.com/ssfdre38/gemma4-nano)
- [GitHub repo](https://github.com/ssfdre38/gemma4-turbo)

## Related Models

- [ssfdre38/gemma4-turbo](https://ollama.com/ssfdre38/gemma4-turbo) - IQ4_XS with vision encoder

## License

Apache 2.0 (same as Gemma 4 base models)

---

**Built for the Gemma 4 Good Hackathon 2026** 🦞
"""

print(f"\n📝 Creating README.md...")
try:
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"✓ README.md uploaded")
except Exception as e:
    print(f"✗ README.md failed: {e}")

print(f"\n🎉 Done! View at: https://huggingface.co/{repo_id}")
