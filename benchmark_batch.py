#!/usr/bin/env python3
"""Test different num_batch values to find the sweet spot for gemma4-turbo:e4b."""

import json
import urllib.request
import subprocess
import time

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "ssfdre38/gemma4-turbo:e4b"

# Common batch sizes to test
BATCH_SIZES = [128, 256, 512, 1024, 2048]

# Use a prompt long enough to stress prompt processing (ingestion speed matters here)
PROMPT = (
    "You are a helpful assistant. "
    "Explain in detail how transformers work, including attention mechanisms, "
    "positional encoding, encoder-decoder architecture, and how they are trained. "
    "Then write a Python class implementing a simple self-attention layer with "
    "full comments explaining each step."
)

def unload_model():
    """Force Ollama to unload the model between runs for a clean state."""
    body = json.dumps({"model": MODEL, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except Exception:
        pass
    time.sleep(2)

def run(batch_size: int):
    body = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {
            "num_predict": 300,
            "num_batch": batch_size,
        },
    }).encode()

    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())

    prompt_tokens  = data.get("prompt_eval_count", 0)
    prompt_ns      = data.get("prompt_eval_duration", 1)
    eval_tokens    = data.get("eval_count", 0)
    eval_ns        = data.get("eval_duration", 1)
    total_ns       = data.get("total_duration", 1)
    load_ns        = data.get("load_duration", 1)

    prompt_tps = prompt_tokens / (prompt_ns / 1e9)
    eval_tps   = eval_tokens   / (eval_ns   / 1e9)

    return {
        "batch":       batch_size,
        "prompt_tps":  prompt_tps,
        "eval_tps":    eval_tps,
        "prompt_tok":  prompt_tokens,
        "eval_tok":    eval_tokens,
        "load_ms":     load_ns / 1e6,
        "total_ms":    total_ns / 1e6,
    }

def main():
    print(f"Benchmarking num_batch on {MODEL}")
    print(f"Prompt: {len(PROMPT.split())} words  |  num_predict: 300")
    print("=" * 68)
    print(f"  {'batch':>6}  {'prompt tok/s':>14}  {'eval tok/s':>12}  {'total ms':>10}")
    print(f"  {'-'*60}")

    results = []
    for bs in BATCH_SIZES:
        print(f"  {bs:>6}  running...", end="", flush=True)
        unload_model()
        try:
            r = run(bs)
            results.append(r)
            print(f"\r  {r['batch']:>6}  {r['prompt_tps']:>14.1f}  {r['eval_tps']:>12.1f}  {r['total_ms']:>10.0f}")
        except Exception as e:
            print(f"\r  {bs:>6}  ERROR: {e}")

    if results:
        best_eval   = max(results, key=lambda x: x["eval_tps"])
        best_prompt = max(results, key=lambda x: x["prompt_tps"])
        print()
        print(f"  Best eval tok/s   → num_batch={best_eval['batch']}   ({best_eval['eval_tps']:.1f} tok/s)")
        print(f"  Best prompt tok/s → num_batch={best_prompt['batch']} ({best_prompt['prompt_tps']:.1f} tok/s)")
        print()
        print(f"  Recommendation: PARAMETER num_batch {best_eval['batch']}")

if __name__ == "__main__":
    main()
