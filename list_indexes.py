#!/usr/bin/env python3
"""
Simple script to list available Pinecone indexes.
"""

import sys
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()


def list_pinecone_indexes():
    """List all available Pinecone indexes"""
    try:
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("❌ PINECONE_API_KEY not found in environment variables")
            return

        pc = Pinecone(api_key=api_key)

        # List indexes
        indexes = pc.list_indexes()

        if not indexes:
            print("❌ No indexes found in your Pinecone project")
            print("🔧 Please create an index first using the Pinecone console")
        else:
            print("✅ Available Pinecone indexes:")
            for i, index_info in enumerate(indexes, 1):
                print(f"   {i}. {index_info['name']}")
                print(f"      - Dimension: {index_info.get('dimension', 'N/A')}")
                print(f"      - Metric: {index_info.get('metric', 'N/A')}")
                print(
                    f"      - Status: {index_info.get('status', {}).get('state', 'N/A')}"
                )
                print()

    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        print("💡 Make sure your PINECONE_API_KEY is correct")


if __name__ == "__main__":
    list_pinecone_indexes()
