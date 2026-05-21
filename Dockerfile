FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including dlib requirements
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dlib first separately
RUN pip install --no-cache-dir dlib

# Copy requirements
COPY requirements.txt .

# Install other packages (without dlib)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p data models plots faces alerts logs

# Expose ports
EXPOSE 5000 8501

CMD ["python3", "start.py"]
