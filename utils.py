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

def format_to_bibtex(citation_json_str: str, filename: str) -> str:
    """Formats a JSON string of citation data into a BibTeX entry."""
    try:
        # --- NEW CLEANING LOGIC ---
        # Use regex to find the JSON object within the raw LLM response,
        # even if it's wrapped in markdown code blocks.
        json_match = re.search(r'\{.*\}', citation_json_str, re.DOTALL)
        
        if not json_match:
            # If no JSON object is found at all, raise an error.
            raise ValueError("No valid JSON object found in the LLM response.")
        
        # Extract the clean JSON string from the match
        clean_json_str = json_match.group(0)
        # --- END OF CLEANING LOGIC ---

        # Now, load the cleaned string
        data = json.loads(clean_json_str)
        
        # --- The rest of the function is the same ---
        match = re.search(r'(\d{4}\.\d{5})', filename)
        arxiv_id = match.group(1) if match else "N/A"
        
        title = data.get("title", "No Title Found")
        authors = " and ".join(data.get("authors", ["N/A"]))
        year = data.get("year", "N/A")
        
        first_author_lastname = authors.split(' ')[-1].lower() if ' ' in authors else "unknown"
        first_title_word = title.split(' ')[0].lower().strip(":") if ' ' in title else "untitled"
        key = f"{first_author_lastname}{year}{first_title_word}"
        
        bibtex_entry = f"""@article{{{key},
          title   = {{{title}}},
          author  = {{{authors}}},
          year    = {{{year}}},
          journal = {{arXiv preprint arXiv:{arxiv_id}}}
        }}"""
        return bibtex_entry

    # Add ValueError to the exceptions we can catch
    except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
        print(f"Error formatting BibTeX: {e}")
        return "Could not generate BibTeX citation. The required data could not be extracted."
