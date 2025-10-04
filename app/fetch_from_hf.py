import os
import numpy as np
import pandas as pd
import requests

# Base URL for raw Hugging Face file access
HF_BASE = "https://huggingface.co/datasets/victor-odunsi/anime-recommender-artifacts/resolve/main"
CACHE_DIR = "/tmp/anime_recommender"  # ✅ writable on Render free tier

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def download_from_hf(filename: str) -> str:
    """
    Download a file from Hugging Face if not already cached locally.
    Returns the local path.
    """
    local_path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(local_path):
        url = f"{HF_BASE}/{filename}"
        print(f"⬇️ Downloading {filename} from Hugging Face…")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved to {local_path}")
    else:
        print(f"✅ Using cached file: {local_path}")
    return local_path


def get_similarity_matrix() -> np.ndarray:
    path = download_from_hf("similarity_matrix.npy")
    return np.load(path, allow_pickle=True)


def get_anime_data() -> pd.DataFrame:
    path = download_from_hf("anime_data.csv")
    return pd.read_csv(path)


def get_trending_anime() -> pd.DataFrame:
    path = download_from_hf("trending_df.csv")
    return pd.read_csv(path)