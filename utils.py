# utils.py (Corrected and Simplified)

import os
import requests
from io import BytesIO
from llama_index.llms.mistralai import MistralAI

def get_llm():
    """
    This function now only prepares and returns the MistralAI client.
    It securely gets the API key from environment variables.
    """
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set. Please set it before running the app.")
    
    return MistralAI(api_key=mistral_api_key, model="mistral-small-latest", timeout=240)

def download_pdf_from_url(url: str):
    """Downloads PDF content from a URL and returns it as a BytesIO stream."""
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None