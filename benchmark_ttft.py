#!/usr/bin/env python3
"""Benchmark Time-To-First-Token (TTFT) with different thread counts and flash attention."""

import json
import time
import urllib.request
import os
import sys

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "ssfdre38/gemma4-turbo:e4b"

# Short prompt (few tokens to prefill) and long prompt (many tokens)
PROMPTS = {
    "short_prefill": "What is 2+2?",
    "medium_prefill": (
        "You are a helpful coding assistant. I have a Python Flask application "
        "with a REST API that handles user authentication, database queries via SQLAlchemy, "
        "and file uploads. The app is slow under load. Here is my main routes file:\n"
        "```python\n@app.route('/upload', methods=['POST'])\ndef upload():\n"
        "    f = request.files['file']\n    f.save(os.path.join(UPLOAD_FOLDER, f.filename))\n"
        "    db.session.add(FileRecord(name=f.filename))\n    db.session.commit()\n    return 'ok'\n```\n"
        "What are the performance bottlenecks and how do I fix them?"
    ),
    "long_prefill": " ".join(["The quick brown fox jumps over the lazy dog."] * 60),  # ~480 tokens
}

def unload():
    body = json.dumps({"model": MODEL, "keep_alive": 0}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r: r.read()
    except: pass
    time.sleep(1)

def measure_ttft_streaming(prompt: str, num_thread: int) -> dict:
    """Measure TTFT using streaming — time until first token arrives."""
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": 50, "num_thread": num_thread},
    }).encode()

    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    t_start = time.perf_counter()
    first_token_time = None
    prompt_eval_ms = None
    eval_tps = None

    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            chunk = json.loads(line.decode())
            if first_token_time is None and chunk.get("response"):
                first_token_time = time.perf_counter() - t_start
            if chunk.get("done"):
                if chunk.get("prompt_eval_duration"):
                    prompt_eval_ms = chunk["prompt_eval_duration"] / 1e6
                if chunk.get("eval_count") and chunk.get("eval_duration"):
                    eval_tps = chunk["eval_count"] / (chunk["eval_duration"] / 1e9)
                break

    return {
        "ttft_ms": round((first_token_time or 0) * 1000, 1),
        "prefill_ms": round(prompt_eval_ms or 0, 1),
        "eval_tps": round(eval_tps or 0, 1),
    }

def run_test(label: str, num_thread: int):
    print(f"\n  [{label}]  num_thread={num_thread}")
    print(f"  {'Prompt':<16} {'TTFT (ms)':>10} {'Prefill (ms)':>13} {'Eval tok/s':>12}")
    print(f"  {'-'*55}")
    results = {}
    for name, prompt in PROMPTS.items():
        unload()
        r = measure_ttft_streaming(prompt, num_thread)
        results[name] = r
        print(f"  {name:<16} {r['ttft_ms']:>10.0f} {r['prefill_ms']:>13.0f} {r['eval_tps']:>12.1f}")
    return results

def main():
    thread_counts = [6, 8, 12]  # physical cores, current e4b setting, all logical
    if len(sys.argv) > 1:
        thread_counts = [int(x) for x in sys.argv[1:]]

    print(f"TTFT Benchmark — {MODEL}")
    print(f"CPU: 6 cores / 12 threads (Xeon E-2236)")
    fa = os.environ.get("OLLAMA_FLASH_ATTENTION", "0")
    print(f"OLLAMA_FLASH_ATTENTION={fa}")
    print("=" * 60)

    all_results = {}
    for t in thread_counts:
        all_results[t] = run_test(f"threads={t}", t)

    # Summary — avg TTFT across prompts
    print(f"\n  SUMMARY — Avg TTFT (ms) across all prompts:")
    print(f"  {'Threads':<10} {'Avg TTFT':>10} {'Avg Eval tok/s':>16}")
    print(f"  {'-'*38}")
    for t, res in all_results.items():
        avg_ttft = sum(r["ttft_ms"] for r in res.values()) / len(res)
        avg_tps  = sum(r["eval_tps"] for r in res.values()) / len(res)
        print(f"  {t:<10} {avg_ttft:>10.0f} {avg_tps:>16.1f}")

if __name__ == "__main__":
    main()
