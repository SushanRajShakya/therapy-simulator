#!/bin/bash

# Therapy Simulator Startup Script

echo "Starting Therapy Simulator..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating from template..."
    cp .env.example .env
    echo "Please edit .env file with your OpenAI API key before continuing."
    echo "Get your API key from: https://platform.openai.com/api-keys"
    exit 1
fi

# Check if OpenAI API key is set
if grep -q "your_openai_api_key_here\|KEY_PLACEHOLDER\|test_key_placeholder" .env; then
    echo "Warning: Please set your OpenAI API key in the .env file"
    echo "Edit .env and replace the placeholder with your actual API key"
    echo "Get your API key from: https://platform.openai.com/api-keys"
    exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting server on http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"

# Start the server
uvicorn server.main:app --reload --port 8000
