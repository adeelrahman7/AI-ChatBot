import requests

print ("Testing ollama connection...")

try:
    response = requests.post('http://localhost:11434/api/generate', 
    json={
        'model': 'llama3.2',
        'prompt': 'Say "Hello, I am ready to help you!" in one sentence',
        'stream': False
    },
    timeout=30
    )
    
    if response.status_code == 200:
        print("Ollama connection successful!")
        print("Ollama says:", response.json()['response'])
    else:
        print(f" Error: Status {response.status_code}")
        
except Exception as e:
    print(f"Ollama connection failed: {e}")

