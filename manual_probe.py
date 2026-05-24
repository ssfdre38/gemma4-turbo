import sys
import os
import requests

OLLAMA_API = "http://127.0.0.1:11434/api/generate"

def test_manual(layer):
    model_name = f"manual-test-L{layer}"
    # Test reasoning
    prompt = "If a train travels 120 miles in 2 hours, what is its average speed? Respond ONLY with the number."
    payload = {"model": model_name, "prompt": prompt, "stream": False, "options": {"num_predict": 10}}
    
    print(f"Executing request for {model_name}...")
    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=60)
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Response: [{resp.json().get('response', '')}]")
        else:
            print(f"Error Response: {resp.text}")
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    # Assuming the previous script failed during request, but model MIGHT be there?
    # Or I need to create it first.
    test_manual(32)
