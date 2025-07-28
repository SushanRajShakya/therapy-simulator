import os
from dotenv import load_dotenv


MODEL_CONFIG = {
    "model": "gpt-4.1-nano",
    "temperature": 0.3,  # Lower temperature for more consistent, professional responses
    "max_tokens": None,  # Let the model decide appropriate response length
}

# Load environment variables
load_dotenv(override=True)

# Configure OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "therapy-cbt")
