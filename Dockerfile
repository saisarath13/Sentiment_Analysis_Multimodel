# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

COPY models /app/models
# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ libffi-dev
RUN python -m spacy download en_core_web_sm
# Install SpaCy model




# Expose Flask port
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
