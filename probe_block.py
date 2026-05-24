import sys
import os
import requests
import time

OLLAMA_API = "http://127.0.0.1:11434/api/generate"

def test_manual_block(start, end):
    model_name = f"manual-block-{start}-{end}"
    print(f"Testing block {start}-{end}...")
    
    # We expect the model to be there from the previous run if we didn't cleanup, 
    # but let's assume we need to create it for a clean test.
    prompt = "If a train travels 120 miles in 2 hours, what is its average speed? Respond ONLY with the number."
    payload = {"model": model_name, "prompt": prompt, "stream": False, "options": {"num_predict": 10}}
    
    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=60)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Response: [{resp.json().get('response', '')}]")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test our previously "Resilient" block at high res
    test_manual_block(30, 34)
