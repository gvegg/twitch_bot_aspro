import requests
import urllib.parse
import random
a = random.randint(0, 50)

prompt = "dog"
params = {
    "width": "1920",
    "height": "1080",
    # "seed": a,
    "model": "turbo",
    "nologo": "true",
    "enhance": "true"	
}
encoded_prompt = urllib.parse.quote(prompt)
url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

try:
    response = requests.get(url, params=params, timeout=300) # Increased timeout for image generation
    response.raise_for_status() # Raise an exception for bad status codes

    with open('generated_image.png', 'wb') as f:
        f.write(response.content)
    print("Image saved as generated_image.png")

except requests.exceptions.RequestException as e:
    print(f"Error fetching image: {e}")
    # Consider checking response.text for error messages from the API
    # if response is not None: print(response.text)