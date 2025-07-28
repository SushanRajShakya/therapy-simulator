#!/usr/bin/env python3
"""
Setup script for therapy simulator RAG engine.
This script initializes the Pinecone vector store with CBT knowledge and therapy datasets.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.rag_engine import RAGEngine


def setup_rag_engine():
    """Initialize RAG engine with CBT knowledge and therapy datasets"""
    print("🚀 Initializing RAG Engine...")

    try:
        # Initialize RAG engine
        rag = RAGEngine()
        print("✅ RAG Engine initialized successfully")

        # Add CBT knowledge base
        print("📚 Adding CBT knowledge base...")
        rag.add_cbt_knowledge_base()
        print("✅ CBT knowledge base added")

        # Load therapy datasets
        print("🔄 Loading therapy datasets...")

        # Dataset 1: Mental health counseling conversations
        try:
            rag.load_therapy_dataset(
                "Amod/mental_health_counseling_conversations", limit=200
            )
            print("✅ Loaded mental health counseling conversations")
        except Exception as e:
            print(f"⚠️ Failed to load counseling conversations: {e}")

        # Dataset 2: Mental health chatbot dataset
        try:
            rag.load_therapy_dataset(
                "heliosbrahma/mental_health_chatbot_dataset", limit=200
            )
            print("✅ Loaded mental health chatbot dataset")
        except Exception as e:
            print(f"⚠️ Failed to load chatbot dataset: {e}")

        # Dataset 3: Mental health conversational data
        try:
            rag.load_therapy_dataset(
                "alexandreteles/mental-health-conversational-data", limit=150
            )
            print("✅ Loaded mental health conversational data")
        except Exception as e:
            print(f"⚠️ Failed to load conversational data: {e}")

        print("\n🎉 RAG Engine setup completed successfully!")
        print("🔍 Testing retrieval...")

        # Test retrieval
        test_queries = [
            "How can I deal with anxiety?",
            "I'm feeling depressed and don't want to do anything",
            "I have negative thoughts about myself",
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            context = rag.retrieve_context(query, k=2)
            for i, ctx in enumerate(context):
                print(f"   {i+1}. {ctx[:150]}...")

        print("\n✅ RAG Engine is ready for use!")

    except Exception as e:
        print(f"❌ Error setting up RAG engine: {e}")
        raise


if __name__ == "__main__":
    setup_rag_engine()
