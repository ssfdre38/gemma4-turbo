"""Benchmark ultra-compressed models vs turbo"""
import requests, time, statistics

OLLAMA_API = "http://127.0.0.1:11434/api/generate"

PROMPTS = {
    'short': "What is the capital of France?",
    'reasoning': "If a train travels 120 miles in 2 hours, what is its average speed?",
    'code': "Write a Python function to reverse a string",
}

MODELS = [
    ('ssfdre38/gemma4-turbo:e2b', 'turbo (IQ4_XS, 4.3GB)'),
    ('gemma4-ultra:e2b', 'ultra (IQ3_M, 3.1GB)'),
    ('gemma4-nano:e2b', 'nano (Q3_K_S, 3.1GB)'),
]

def benchmark_model(model_name, prompt, num_predict=200):
    """Time a single generation"""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict}
    }
    start = time.perf_counter()
    resp = requests.post(OLLAMA_API, json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    
    if resp.status_code != 200:
        return None
    data = resp.json()
    tokens = data.get('eval_count', 0)
    tok_s = tokens / elapsed if elapsed > 0 else 0
    return {'time': elapsed, 'tokens': tokens, 'tok_s': tok_s}

def run_benchmarks():
    results = {}
    
    for model_name, label in MODELS:
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {label}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")
        
        results[model_name] = {}
        
        for prompt_type, prompt in PROMPTS.items():
            print(f"  [{prompt_type}] ", end='', flush=True)
            
            runs = []
            for i in range(3):
                result = benchmark_model(model_name, prompt)
                if result:
                    runs.append(result)
                    print(f"{result['tok_s']:.1f} tok/s ", end='', flush=True)
            
            if runs:
                avg_time = statistics.mean(r['time'] for r in runs)
                avg_toks = statistics.mean(r['tok_s'] for r in runs)
                results[model_name][prompt_type] = {
                    'time': avg_time,
                    'tok_s': avg_toks
                }
                print(f"→ avg: {avg_toks:.1f} tok/s")
            else:
                print("FAILED")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("  RESULTS SUMMARY")
    print(f"{'='*80}")
    
    turbo_name = 'ssfdre38/gemma4-turbo:e2b'
    for prompt_type in PROMPTS.keys():
        print(f"\n  {prompt_type.upper()}:")
        turbo_speed = results[turbo_name][prompt_type]['tok_s']
        
        for model_name, label in MODELS:
            speed = results[model_name][prompt_type]['tok_s']
            speedup = speed / turbo_speed if turbo_speed > 0 else 0
            print(f"    {label:30s}: {speed:6.1f} tok/s  ({speedup:4.2f}x)")
    
    # Overall averages
    print(f"\n  OVERALL AVERAGE:")
    for model_name, label in MODELS:
        avg_speed = statistics.mean(
            results[model_name][pt]['tok_s'] for pt in PROMPTS.keys()
        )
        turbo_avg = statistics.mean(
            results[turbo_name][pt]['tok_s'] for pt in PROMPTS.keys()
        )
        speedup = avg_speed / turbo_avg if turbo_avg > 0 else 0
        print(f"    {label:30s}: {avg_speed:6.1f} tok/s  ({speedup:4.2f}x)")

if __name__ == '__main__':
    run_benchmarks()
