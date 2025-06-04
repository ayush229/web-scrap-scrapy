# Use a slim Python base image (e.g., Debian bookworm)
FROM python:3.12-slim-bookworm

# Set environment variables for the virtual environment and path
ENV NIXPACKS_PATH=/opt/venv/bin:$NIXPACKS_PATH
# Ensures Python output is unbuffered, good for logs
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies required for building Python packages like Scrapy and Twisted
# This step should run before installing Python dependencies from requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container first to leverage Docker's build cache
COPY requirements.txt .

# Create and activate a virtual environment, then install Python dependencies
# Using --no-cache-dir for pip to ensure minimal image size
RUN python -m venv --copies /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app

# Set the command to run your Flask application using Gunicorn (shell form for $PORT expansion)
CMD /opt/venv/bin/gunicorn --bind 0.0.0.0:$PORT main:app
