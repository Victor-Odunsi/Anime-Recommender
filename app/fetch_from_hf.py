import os
import numpy as np
import pandas as pd
import requests
from io import BytesIO

# Base URL for raw Hugging Face file access
HF_BASE = "https://huggingface.co/datasets/victor-odunsi/anime-recommender-artifacts/resolve/main"

def download_from_hf(filename: str) -> BytesIO:
    """
    Always download a file directly from Hugging Face.
    Returns a BytesIO stream (no caching).
    """
    url = f"{HF_BASE}/{filename}"
    print(f"⬇️ Downloading {filename} from Hugging Face…")
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)

def get_similarity_matrix() -> pd.DataFrame:
    file_path = download_from_hf("similarity_matrix.csv")
    return pd.read_csv(file_path)

def get_anime_data() -> pd.DataFrame:
    file_path = download_from_hf("anime_data.csv")
    return pd.read_csv(file_path)

def get_trending_anime() -> pd.DataFrame:
    file_path = download_from_hf("trending_df.csv")
    return pd.read_csv(file_path)
