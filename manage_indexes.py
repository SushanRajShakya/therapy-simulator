#!/usr/bin/env python3
"""
Pinecone Index Management Script
This script helps you manage your Pinecone indexes for the therapy simulator.
"""

import sys
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time

# Load environment variables
load_dotenv()


def get_pinecone_client():
    """Get Pinecone client"""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found in environment variables")
        sys.exit(1)
    return Pinecone(api_key=api_key)


def list_indexes(pc):
    """List all available Pinecone indexes"""
    try:
        indexes = pc.list_indexes()

        if not indexes:
            print("❌ No indexes found in your Pinecone project")
        else:
            print("✅ Available Pinecone indexes:")
            for i, index_info in enumerate(indexes, 1):
                print(f"   {i}. {index_info['name']}")
                print(f"      - Dimension: {index_info.get('dimension', 'N/A')}")
                print(f"      - Metric: {index_info.get('metric', 'N/A')}")
                print(
                    f"      - Status: {index_info.get('status', {}).get('state', 'N/A')}"
                )
                if "spec" in index_info:
                    spec = index_info["spec"]
                    if hasattr(spec, "serverless"):
                        print(f"      - Cloud: {spec.serverless.cloud}")
                        print(f"      - Region: {spec.serverless.region}")
                print()

        return indexes
    except Exception as e:
        print(f"❌ Error listing indexes: {e}")
        return []


def delete_index(pc, index_name):
    """Delete a Pinecone index"""
    try:
        print(f"🗑️  Deleting index: {index_name}")
        pc.delete_index(index_name)
        print(f"✅ Successfully deleted index: {index_name}")
        return True
    except Exception as e:
        print(f"❌ Error deleting index {index_name}: {e}")
        return False


def create_index(
    pc, index_name, dimension=1536, metric="cosine", cloud="aws", region="us-east-1"
):
    """Create a new Pinecone index"""
    try:
        print(f"🚀 Creating new index: {index_name}")
        print(f"   - Dimension: {dimension}")
        print(f"   - Metric: {metric}")
        print(f"   - Cloud: {cloud}")
        print(f"   - Region: {region}")

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

        # Wait for index to be ready
        print("⏳ Waiting for index to be ready...")
        while True:
            try:
                status = pc.describe_index(index_name).status["ready"]
                if status:
                    break
                time.sleep(1)
            except:
                time.sleep(1)

        print(f"✅ Successfully created index: {index_name}")
        return True
    except Exception as e:
        print(f"❌ Error creating index {index_name}: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print(
            """
🔧 Pinecone Index Management Script

Usage:
  python manage_indexes.py list                    - List all indexes
  python manage_indexes.py delete <index_name>     - Delete an index
  python manage_indexes.py create <index_name>     - Create new index (1536 dims)
  python manage_indexes.py recreate <index_name>   - Delete and recreate index

Examples:
  python manage_indexes.py list
  python manage_indexes.py delete therapy-simulator
  python manage_indexes.py create therapy-simulator-1536
  python manage_indexes.py recreate therapy-simulator
"""
        )
        sys.exit(1)

    pc = get_pinecone_client()
    command = sys.argv[1].lower()

    if command == "list":
        list_indexes(pc)

    elif command == "delete":
        if len(sys.argv) < 3:
            print("❌ Please provide index name to delete")
            sys.exit(1)
        index_name = sys.argv[2]
        delete_index(pc, index_name)

    elif command == "create":
        if len(sys.argv) < 3:
            print("❌ Please provide index name to create")
            sys.exit(1)
        index_name = sys.argv[2]
        create_index(pc, index_name)

    elif command == "recreate":
        if len(sys.argv) < 3:
            print("❌ Please provide index name to recreate")
            sys.exit(1)
        index_name = sys.argv[2]

        # Check if index exists first
        indexes = list_indexes(pc)
        index_exists = any(idx["name"] == index_name for idx in indexes)

        if index_exists:
            print(f"⚠️  Index '{index_name}' exists. Deleting it first...")
            if delete_index(pc, index_name):
                print("⏳ Waiting a moment before creating new index...")
                time.sleep(5)  # Wait for deletion to propagate
            else:
                print("❌ Failed to delete existing index")
                sys.exit(1)

        create_index(pc, index_name)

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
