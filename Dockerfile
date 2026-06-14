# ECG Heartbeat Classifier - container image (CPU build for portability).
# For GPU, run with `--gpus all` and install a CUDA torch build instead.
FROM python:3.11-slim

WORKDIR /app

# System deps occasionally needed by matplotlib/scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the source (data/ and saved_models/ are mounted at runtime).
COPY . .

EXPOSE 8501

# Default: launch the interactive Streamlit demo.
# Override to train, e.g.:  docker run --rm -v ${PWD}/data:/app/data <img> python main.py train
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
