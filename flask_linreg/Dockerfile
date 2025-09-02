# Use the official Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies that might be needed for numpy/scipy
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory structure expected by your Flask app
RUN mkdir -p flask_linreg/data flask_linreg/artifacts

# RUN train_model.py to create and save the model 
COPY train_model.py .
RUN python train_model.py

# Copy your application code
COPY . .

EXPOSE 5000

# Use the correct module name (app refers to app.py, the second app is the Flask instance)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]