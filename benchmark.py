#!/usr/bin/env python3
"""Benchmark gemma4-turbo vs base gemma4 — tokens/sec comparison."""

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

PAIRS = [
    ("gemma4:e2b",                   "ssfdre38/gemma4-turbo:e2b"),
    ("gemma4:e4b",                   "ssfdre38/gemma4-turbo:e4b"),
    ("gemma4:26b",                   "ssfdre38/gemma4-turbo:26b"),
    ("gemma4:31b",                   "ssfdre38/gemma4-turbo:31b"),
]

PROMPTS = [
    {
        "name": "short",
        "text": "What is the capital of France? Answer in one sentence.",
    },
    {
        "name": "reasoning",
        "text": "Explain step by step how a neural network learns using backpropagation.",
    },
    {
        "name": "code",
        "text": "Write a Python function that checks if a number is prime, with a docstring and type hints.",
    },
]

@dataclass
class Result:
    model: str
    prompt: str
    prompt_tokens: int
    eval_tokens: int
    prompt_ms: float
    eval_ms: float
    tok_per_sec: float
    text_preview: str

def run(model: str, prompt: str) -> Result:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 200},
    }).encode()

    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - t0

    eval_count   = data.get("eval_count", 0)
    eval_dur_ns  = data.get("eval_duration", 1)
    prompt_count = data.get("prompt_eval_count", 0)
    prompt_dur_ns= data.get("prompt_eval_duration", 1)

    tok_per_sec = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns else 0

    return Result(
        model=model,
        prompt=prompt[:40] + "...",
        prompt_tokens=prompt_count,
        eval_tokens=eval_count,
        prompt_ms=prompt_dur_ns / 1e6,
        eval_ms=eval_dur_ns / 1e6,
        tok_per_sec=tok_per_sec,
        text_preview=(data.get("response", "")[:80]).replace("\n", " "),
    )

def benchmark_pair(base: str, turbo: str, prompt_name: str, prompt: str):
    print(f"\n  [{prompt_name}]  base={base}  turbo={turbo}")
    print(f"  Running {base}...", end="", flush=True)
    r_base = run(base, prompt)
    print(f" {r_base.tok_per_sec:.1f} tok/s  ({r_base.eval_tokens} tokens)")

    print(f"  Running {turbo}...", end="", flush=True)
    r_turbo = run(turbo, prompt)
    print(f" {r_turbo.tok_per_sec:.1f} tok/s  ({r_turbo.eval_tokens} tokens)")

    speedup = r_turbo.tok_per_sec / r_base.tok_per_sec if r_base.tok_per_sec else 0
    size_diff = ""
    return r_base, r_turbo, speedup

def main():
    import sys
    # Allow filtering: python benchmark.py e4b
    filter_size = sys.argv[1] if len(sys.argv) > 1 else None

    all_results = []
    for base, turbo in PAIRS:
        size = base.split(":")[-1]
        if filter_size and size != filter_size:
            continue

        print(f"\n{'='*60}")
        print(f"  {size.upper()}: {base}  vs  {turbo}")
        print(f"{'='*60}")

        size_results = []
        for p in PROMPTS:
            try:
                r_base, r_turbo, speedup = benchmark_pair(base, turbo, p["name"], p["text"])
                size_results.append((p["name"], r_base, r_turbo, speedup))
            except Exception as e:
                print(f"  ERROR: {e}")

        if size_results:
            print(f"\n  Summary for {size.upper()}:")
            print(f"  {'Prompt':<12} {'Base tok/s':>12} {'Turbo tok/s':>12} {'Speedup':>10}")
            print(f"  {'-'*48}")
            speedups = []
            for name, rb, rt, sp in size_results:
                speedups.append(sp)
                marker = "🚀" if sp > 1.1 else ("🐢" if sp < 0.9 else "≈")
                print(f"  {name:<12} {rb.tok_per_sec:>12.1f} {rt.tok_per_sec:>12.1f} {sp:>9.2f}x {marker}")
            avg = sum(speedups) / len(speedups)
            print(f"  {'Average':<12} {'':>12} {'':>12} {avg:>9.2f}x")
            all_results.append((size, speedups, avg))

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Size':<8} {'Avg Speedup':>12}")
        for size, _, avg in all_results:
            marker = "🚀" if avg > 1.1 else ("🐢" if avg < 0.9 else "≈")
            print(f"  {size:<8} {avg:>11.2f}x {marker}")

if __name__ == "__main__":
    main()
