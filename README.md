# 🎥 Anime Recommendation System

A fun **content-based recommendation system** that combines my love for anime with data science.  
The app recommends similar anime based on **synopsis, genres, and themes** using **TF-IDF vectorization** and **cosine similarity**.

🔗 **Live Demo**: [anime-recommender-tld1.onrender.com](https://anime-recommender-tld1.onrender.com)

---

## ✨ Features
- 📖 **Content-based recommendations** from anime synopsis, genres, and themes  
- 🔄 **Weekly dataset updates** from [Jikan](https://jikan.moe/)
- 🗂 **Persistent storage** for dataset and embeddings on Render  
- 🎨 **Interactive Streamlit UI** to explore and discover new anime  

---

## 🛠 Tech Stack
- **Python**, **pandas**, **scikit-learn**
- **Streamlit** for the user interface  
- **Render** for deployment + persistent disk  
- **Cron Jobs** for automated weekly updates  
- **Jikan / AniList APIs** for fresh anime data  
