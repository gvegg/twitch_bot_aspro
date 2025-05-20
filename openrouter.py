import requests

url = "https://text.pollinations.ai/models"

try:
    response = requests.get(url)
    response.raise_for_status()
    models = response.json()
    print("Available Image Models:")
    for model in models:
        print(f"- {model}")
except requests.exceptions.RequestException as e:
    print(f"Error fetching models: {e}")