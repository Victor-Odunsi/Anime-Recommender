#!/bin/bash
set -e

# Ensure /data exists
mkdir -p /data

# If /data is empty, seed it with artifacts
if [ ! -f /data/anime_data.csv ]; then
    echo "📥 Seeding /data with base artifacts..."
    cp -r /app/artifacts/* /data/
fi

# Start Streamlit
exec streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
