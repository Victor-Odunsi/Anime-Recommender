import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
import re, nltk
from nltk.stem import PorterStemmer
from datetime import datetime

logging.basicConfig(filename = 'app.log', level=logging.INFO)

Base_dir = Path(__file__).resolve().parent.parent

DATA_PATH = Base_dir / 'artifacts' / 'anime_data.csv'
SIM_PATH = Base_dir / 'artifacts' / 'similarity_matrix.npy'
TRENDING_PATH = Base_dir / 'artifacts' / 'trending_df.csv'

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
            'themes': [t["name"] for t in item.get("themes", [])],
            'demographics': [d["name"] for d in item.get("demographics", [])],
            'synopsis': item.get('synopsis', ''),
            'type': item.get('type', ''),
            'episodes': item.get("episodes", None),
            'producers': [p["name"] for p in item.get("producers", [])],
            'source': item.get("source", ""),
            'combined_features': None
        })
    
    new_df = pd.DataFrame(new_anime_list)

    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH)
    else:
        existing_df = pd.DataFrame()

    combined_df = (
        pd.concat([existing_df, new_df])
        .drop_duplicates(subset=['anime_id'])
        .reset_index(drop=True)
    )

    trending_df = new_df.sort_values(by='score', ascending=False).head()
    
    combined_df.to_csv(DATA_PATH, index=False)
    logging.info(
        f"Dataset updated with {len(new_df)} new entries today {datetime.now().strftime("%d-%m-%Y")}."
         "Total entries: {len(combined_df)}"
        )
    trending_df.to_csv(TRENDING_PATH, index=False)
    logging.info(f"Trending dataset updated today {datetime.now().strftime("%d-%m-%Y")}.")


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

def compute_similalrity_matrix():

    if not DATA_PATH.exists():
        logging.error("Data file not found. Please run update_dataset first.")
        return
    
    anime_data = pd.read_csv(DATA_PATH)

    selected_features = [
        'genres',
        'themes', 
        'demographics', 
        'synopsis', 
        'type',
        'producers', 
        'source'
    ]
    for feature in selected_features:
        anime_data[feature] = anime_data[feature].fillna('')

    anime_data['combined_features'] = (
        anime_data['genres'] + ' ' +
        anime_data['themes'] + ' ' +
        anime_data['demographics'] + ' ' +
        anime_data['synopsis'] + ' ' +
        anime_data['type'] + ' ' +
        anime_data['producers'] + ' ' +
        anime_data['source']
    )

    anime_data['combined_features'] = (
        anime_data['combined_features']
        .apply(_preprocess)
    )

    tfidf = TfidfVectorizer(stop_words='english')
    feature_matrix = tfidf.fit_transform(anime_data['combined_features'])

    similarity = cosine_similarity(feature_matrix)

    np.save(SIM_PATH, similarity.astype("float32"))
    logging.info(
        f"Similarity matrix computed and saved today {datetime.now().strftime("%d-%m-%Y")}."
        )
    

update_dataset()
compute_similalrity_matrix()