#!/bin/bash

# Therapy Simulator Frontend Startup Script

echo "Starting Therapy Simulator Frontend..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating from template..."
    cp .env.example .env
    echo "Please edit .env file with your OpenAI API key before continuing."
    exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting frontend on http://localhost:8501"
echo "Make sure the server is running on port 8000"
echo "Press Ctrl+C to stop the frontend"

# Start the frontend
streamlit run frontend/main.py
