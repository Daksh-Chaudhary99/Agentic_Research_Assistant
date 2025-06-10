import os
import json
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
    
    return MistralAI(api_key=mistral_api_key, model="mistral-medium-latest", timeout=240)

def download_pdf_from_url(url: str):
    """Downloads PDF content from a URL and returns it as a BytesIO stream."""
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def format_to_bibtex(citation_json_str: str, arxiv_id: str) -> str:
    """Formats a JSON string of citation data into a BibTeX entry."""
    try:
        data = json.loads(citation_json_str)
        title = data.get("title", "No Title Found")
        authors = " and ".join(data.get("authors", ["N/A"]))
        year = data.get("year", "N/A")
        
        # Create a simple citation key, e.g., "bouzenia2024"
        first_author_lastname = authors.split(' ')[-1].lower() if ' ' in authors else authors.lower()
        key = f"{first_author_lastname}{year}"
        
        bibtex_entry = f"""@article{{{key},
            title   = {{{title}}},
            author  = {{{authors}}},
            year    = {{{year}}},
            journal = {{arXiv preprint arXiv:{arxiv_id}}}
        }}"""
        return bibtex_entry
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error formatting BibTeX: {e}")
        return "Could not generate BibTeX citation. The required data could not be extracted."