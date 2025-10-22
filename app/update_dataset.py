import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging, re, os, pickle
from nltk.stem import PorterStemmer
from datetime import datetime
from io import BytesIO
from huggingface_hub import upload_file
from fetch_from_hf import get_anime_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Persistent path
BASE_PATH = Path("artifacts")
BASE_PATH.mkdir(exist_ok=True)
HF_DATA_BASE = "https://huggingface.co/datasets/victor-odunsi/anime-recommender-artifacts/resolve/main"
HF_REPO = "victor-odunsi/anime-recommender-artifacts"
HF_TOKEN = os.getenv("HF_TOKEN")  

DATA_PATH = BASE_PATH / "anime_data.csv"
TRENDING_PATH = BASE_PATH / "trending_df.csv"
MATRIX_PATH = BASE_PATH / "similarity_matrix.csv"

def update_dataset():
    try:
        url = "https://api.jikan.moe/v4/seasons/now?limit=25"
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        logging.error(f"Error fetching data from Jikan API: {e}")
        return

    if "data" not in data:
        logging.error("Unexpected response structure from Jikan API")
        return None

    new_anime_list = []
    for item in data['data']:
        new_anime_list.append({
            'anime_id': item['mal_id'],
            'anime_url': item['url'],
            'image_url': item['images']['jpg']['image_url'],
            'name': item['title'],
            'score': item.get('score', None),
            'themes': " ".join([t["name"] for t in item.get("themes", [])]),
            'demographics': " ".join([d["name"] for d in item.get("demographics", [])]),
            'synopsis': item.get('synopsis', ''),
            'type': item.get('type', ''),
            'episodes': item.get("episodes", None),
            'producers': " ".join([p["name"] for p in item.get("producers", [])]),
            'source': item.get("source", ""),
            'combined_features': ""
        })

    new_df = pd.DataFrame(new_anime_list)

    existing_df = get_anime_data()

    combined_df = (
        pd.concat([existing_df, new_df])
        .drop_duplicates(subset=['anime_id'])
        .reset_index(drop=True)
    )

    trending_df = new_df.sort_values(by='score', ascending=False).head()

    combined_df.to_csv(DATA_PATH, index=False)
    trending_df.to_csv(TRENDING_PATH, index=False)

    logging.info(
        f"Dataset updated with {len(new_df)} new entries today {datetime.now().strftime('%d-%m-%Y')}."
        f" Total entries: {len(combined_df)}"
    )

    return combined_df

def _preprocess(text):
    if not isinstance(text, str):
        return ""
    ps = PorterStemmer()
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', '', text)
    text = " ".join(ps.stem(word) for word in text.split())
    return text

def compute_recommendations(df: pd.DataFrame):
    """Compute nearest neighbors and save as CSV"""
    df = df.fillna("")
    df['combined_features'] = (
        df['themes'] + " " +
        df['demographics'] + " " +
        df['synopsis'] + " " +
        df['type'] + " " +
        df['producers'] + " " +
        df['source']
    ).apply(_preprocess)

    tfidf = TfidfVectorizer(stop_words='english')
    feature_matrix = tfidf.fit_transform(df['combined_features'])

    similarity = cosine_similarity(feature_matrix)

    anime_dict = {}
    for i in range(similarity.shape[0]):
        top_idx = np.argsort(similarity[i])[-11:][::-1]
        top_idx = [idx for idx in top_idx if idx != i]
        anime_dict[i] = top_idx[:10]

    recs_df = pd.DataFrame.from_dict(anime_dict, orient='index')
    recs_df.to_csv(MATRIX_PATH, index=False)

    logging.info(f"Recommendations precomputed on {datetime.now().strftime('%d-%m-%Y')}.")


def upload_to_hf():
    """Upload CSVs to Hugging Face dataset repo"""
    files = [DATA_PATH, TRENDING_PATH, MATRIX_PATH]

    for f in files:
        filename = f.name
        print(f"⬆️ Uploading {f} to {HF_REPO}...")
        upload_file(
            path_or_fileobj=str(f),
            path_in_repo=filename,
            repo_id=HF_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
    print("✅ All files uploaded successfully!")

    logging.info(f"Similarity matrix computed and saved today {datetime.now().strftime('%d-%m-%Y')}.")

if __name__ == "__main__":
    df = update_dataset()
    if df is not None:
        compute_recommendations(df)
        upload_to_hf()