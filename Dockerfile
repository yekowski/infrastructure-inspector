FROM python:3.11-slim

# Install system libraries required by OpenCV (libgl1 and libglib2.0-0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory inside the container
WORKDIR /app

# Create a system user and group (appuser:appgroup) to run the app as non-root
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Pre-install CPU-only PyTorch to bypass heavy GPU/CUDA downloads
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install dependencies (utilizing headless OpenCV build from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Adjust folder ownership to the non-root appuser
RUN chown -R appuser:appgroup /app

# Switch executing user context to non-root
USER appuser

# Expose primary Vision CLI script as default ENTRYPOINT
ENTRYPOINT ["python3", ".agents/skills/inspector/scripts/analyze_image.py"]
