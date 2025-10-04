FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and configs
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

CMD ["bash", "-c", "streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0"]