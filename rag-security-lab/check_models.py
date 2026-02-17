import os

import google.genai as genai
from dotenv import load_dotenv

# Assumes .env is in the parent directory
load_dotenv(dotenv_path="../.env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Could not find GOOGLE_API_KEY in the .env file.")

genai.configure(api_key=api_key)

print("--- Available Models for Embedding ---")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)
print("------------------------------------")
