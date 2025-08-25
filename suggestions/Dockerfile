# Use Python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements first (for caching layers)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into container
COPY . .

# Expose port Flask will run on
EXPOSE 5000

# Run the main file
CMD ["python", "WordSuggestion.py"]
