# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g., for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port from FastAPI (default 8000)
EXPOSE 8000

# Command to run the application
# Assuming main.py is in 'api' folder or root. 
# Based on 'api/main.py' existence in list_dir, and structure.
# If main.py is in 'api' folder, uvicorn must be pointed correctly.
# 'api.main:app' suggests running from root.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]