import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import time
from fetch_from_hf import get_anime_data, get_similarity_matrix, get_trending_anime

# Page configuration
st.set_page_config(
    page_title="AnimeMatch",
    page_icon="🎬",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* Title and subtitle */
    .title-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .main-title {
        font-size: 4rem;
        font-weight: bold;
        background: linear-gradient(45deg, #8b5cf6, #ec4899, #f59e0b, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-family: 'Arial', sans-serif;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.5;
    }

    /* Search box */
    .stTextInput > div > div > input {
        background: rgba(30, 30, 30, 0.8);
        border: 1px solid #374151;
        border-radius: 12px;
        color: white;
        font-size: 1.1rem;
        padding: 1rem 1rem 1rem 3rem;
        height: 60px;
    }

    .stTextInput > div > div > input::placeholder {
        color: #6b7280;
    }

    /* Recommend button */
    .recommend-button {
        background: linear-gradient(135deg, #8b5cf6, #ec4899);
        border: none;
        border-radius: 12px;
        color: white;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
        height: 60px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Movies grid */
    .movies-grid {
        display: flex;
        gap: 1.5rem;
        overflow-x: auto;
        padding: 1rem 0;
    }

    .movie-card {
        flex: 0 0 250px;
        background: rgba(30, 30, 30, 0.8);
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #374151;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
    }

    .movie-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        border-color: #8b5cf6;
    }

    .movie-poster {
        width: 100%;
        height: 300px;
        object-fit: cover;
    }

    .movie-info {
        padding: 1.5rem;
    }

    .movie-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
        line-height: 1.3;
        text-align: center;
    }

    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Custom scrollbar */
    .movies-grid::-webkit-scrollbar {
        height: 8px;
    }

    .movies-grid::-webkit-scrollbar-track {
        background: rgba(30, 30, 30, 0.5);
        border-radius: 4px;
    }

    .movies-grid::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8b5cf6, #ec4899);
        border-radius: 4px;
    }

    .movies-grid::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7c3aed, #db2777);
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="title-container">
    <div class="main-title">🎬 AnimeMatch</div>
    <div class="subtitle">
        Discover your next favorite anime with ML-powered recommendations tailored just for you
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_data(ttl = 259200)
def _get_anime_data():
    return get_anime_data()

@st.cache_data(ttl = 259200)
def _get_similarity_matrix():
    return get_similarity_matrix()

@st.cache_data(ttl = 259200)
def _get_trending_anime():
    return get_trending_anime()

@st.cache_resource
def _load_image(url: str):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Load artifacts
trending_df = _get_trending_anime()
anime_data = _get_anime_data()
similarity_df = _get_similarity_matrix()

def recommend(anime):
    anime_index = anime_data[anime_data['name'] == anime].index[0]
    neighbor_ids = similarity_df.loc[anime_index].dropna().astype(int).tolist()

    candidates = []
    for idx in neighbor_ids[:8]:  # Limit to top 8 recommendations
        candidates.append(
            {
                'name': anime_data.iloc[idx]['name'],
                'poster_url': anime_data.iloc[idx]['image_url'],
                'anime_url': anime_data.iloc[idx]['anime_url'],
                'score': anime_data.iloc[idx]['score']
            }
        )
    
    candidates = sorted(candidates, reverse=True, key=lambda x: x['score'])

    recommended_anime_names = [c['name'] for c in candidates]
    recommended_anime_posters = [c['poster_url'] for c in candidates]
    recommended_anime_urls = [c['anime_url'] for c in candidates]

    return recommended_anime_names, recommended_anime_posters, recommended_anime_urls

anime_list = anime_data['name'].values

search_query = st.selectbox(
    '🎯 Enter your movie preferences',
    anime_list
)

if st.button("✨ Get Recommendations"):
    with st.spinner('Finding the best matches for you...'):
        time.sleep(2)  # ⏳ Add artificial delay (1–2 seconds)

        recommended_names, recommended_posters, recommended_urls = recommend(search_query)
        
        cols = st.columns(len(recommended_names))
        for idx, col in enumerate(cols):
            with col:
                col.image(
                    recommended_posters[idx],
                    use_container_width=True,
                    caption=recommended_names[idx]
                )
                col.markdown(
                    f"""
                    <a href="{recommended_urls[idx]}" target="_blank">
                        <button style="margin-top:5px; padding:0.4em 1em; border-radius:8px; border:none; background-color:#4CAF50; color:white; cursor:pointer;">
                            🔗 More Info
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True
                )


st.subheader("🔥 Trending Anime Right Now")
cols = st.columns(len(trending_df))
for idx, col in enumerate(cols):
    with col:
        img = _load_image(trending_df.iloc[idx]['image_url'])
        st.image(
            img,
            use_container_width=True,
            caption=trending_df.iloc[idx]['name']
            )
        st.write(
            f"{trending_df.iloc[idx]['synopsis'][:100]}..." 
            if trending_df.iloc[idx]['synopsis'] else ""
        )
        col.markdown(
                    f"""
                    <a href="{trending_df.iloc[idx]['anime_url']}" target="_blank">
                        <button style="margin-top:5px; padding:0.4em 1em; border-radius:8px; border:none; background-color:#4CAF50; color:white; cursor:pointer;">
                            🔗 More Info
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True
                )
        
st.markdown("""
<div style="margin-top: 4rem; text-align: center; color: #6b7280; padding: 2rem;">
    <p>🎬 Discover • Watch • Enjoy</p>
</div>
""", unsafe_allow_html=True)