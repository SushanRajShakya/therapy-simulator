import os
from typing import List, Dict, Optional
from pinecone import Pinecone
from datasets import load_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever

from server.config import *


class RAGEngine:
    def __init__(self):
        """Initialize RAG Engine with Pinecone vector store"""
        self.index_name = PINECONE_INDEX_NAME
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",  # Latest embedding model
            api_key=OPENAI_API_KEY,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=MODEL_CONFIG["model"],
            temperature=MODEL_CONFIG["temperature"],
        )

        # Initialize vector store
        self._setup_index()
        self.vectorstore = PineconeVectorStore(
            index=self.index, embedding=self.embeddings, text_key="text"
        )

    def _setup_index(self):
        """Connect to existing Pinecone index"""
        self.index = self.pc.Index(self.index_name)

    def add_documents(
        self, documents: List[str], metadatas: Optional[List[Dict]] = None
    ):
        """Add documents to the vector store"""
        docs = []
        for i, doc_text in enumerate(documents):
            chunks = self.text_splitter.split_text(doc_text)
            for chunk in chunks:
                metadata = metadatas[i] if metadatas else {}
                docs.append(Document(page_content=chunk, metadata=metadata))

        if docs:
            self.vectorstore.add_documents(docs)
            print(f"Added {len(docs)} document chunks to vector store")

    def add_cbt_knowledge_base(self):
        """Add CBT-specific knowledge to the vector store"""
        cbt_techniques = [
            {
                "text": """ABC Model (Activating Event, Belief, Consequence): This foundational CBT technique helps identify the connection between situations, thoughts, and emotional responses. The Activating Event is the trigger situation, the Belief is the thought or interpretation about the event, and the Consequence is the emotional and behavioral response. By examining these three components, clients can identify how their interpretations of events (rather than the events themselves) create their emotional distress. This technique is particularly effective for anxiety, depression, and anger management.""",
                "metadata": {
                    "technique": "ABC Model",
                    "category": "cognitive restructuring",
                },
            },
            {
                "text": """Cognitive Restructuring: This technique involves identifying and challenging negative thought patterns and cognitive distortions. Common distortions include all-or-nothing thinking, catastrophizing, mind reading, and overgeneralization. The process involves: 1) Identifying the negative thought, 2) Examining evidence for and against the thought, 3) Developing balanced, realistic alternative thoughts, 4) Testing these new thoughts behaviorally. This technique is central to treating depression, anxiety disorders, and low self-esteem.""",
                "metadata": {
                    "technique": "Cognitive Restructuring",
                    "category": "cognitive restructuring",
                },
            },
            {
                "text": """Behavioral Activation: This technique focuses on increasing engagement in meaningful, pleasurable, or mastery-oriented activities. It's based on the principle that behavior influences mood. The process involves: identifying values and goals, scheduling pleasant activities, monitoring mood changes, and gradually increasing activity levels. Behavioral activation is particularly effective for depression, as it helps break the cycle of withdrawal and inactivity that maintains depressive symptoms.""",
                "metadata": {
                    "technique": "Behavioral Activation",
                    "category": "behavioral intervention",
                },
            },
            {
                "text": """Exposure Therapy: A behavioral technique used primarily for anxiety disorders, phobias, and PTSD. It involves gradual, controlled exposure to feared situations or objects in a safe environment. The exposure can be imaginal (visualizing the feared situation) or in vivo (real-life exposure). The process helps clients learn that their feared consequences are unlikely to occur and that anxiety naturally decreases over time. Systematic desensitization and graded exposure hierarchies are common variations.""",
                "metadata": {
                    "technique": "Exposure Therapy",
                    "category": "behavioral intervention",
                },
            },
            {
                "text": """Mindfulness and Acceptance Strategies: These techniques, borrowed from third-wave CBT approaches, help clients observe thoughts and feelings without judgment. Mindfulness practices include breathing exercises, body scans, and present-moment awareness. Acceptance strategies involve acknowledging difficult emotions without trying to change them immediately. These techniques are effective for anxiety, depression, chronic pain, and emotional regulation difficulties.""",
                "metadata": {
                    "technique": "Mindfulness",
                    "category": "acceptance-based",
                },
            },
            {
                "text": """Problem-Solving Therapy: A structured approach to addressing specific life problems that contribute to emotional distress. The steps include: 1) Problem identification and definition, 2) Goal setting, 3) Brainstorming solutions, 4) Evaluating pros and cons of each solution, 5) Implementing the chosen solution, 6) Evaluating outcomes. This technique is useful for clients facing concrete life challenges alongside their emotional difficulties.""",
                "metadata": {
                    "technique": "Problem-Solving",
                    "category": "behavioral intervention",
                },
            },
        ]

        documents = [item["text"] for item in cbt_techniques]
        metadatas = [item["metadata"] for item in cbt_techniques]
        self.add_documents(documents, metadatas)

    def load_mental_health_conversations(self, limit: Optional[int] = 300):
        """Load the specific mental health counseling conversations dataset"""
        dataset_name = "Amod/mental_health_counseling_conversations"
        try:
            print(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split="train")

            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))

            documents = []
            metadatas = []

            for item in dataset:
                # This dataset typically has 'Context' and 'Response' fields
                # We'll combine them to create meaningful therapy examples
                context = item.get("Context", "")
                response = item.get("Response", "")

                if context and response:
                    # Create a conversation format
                    conversation_text = f"Client: {context}\nTherapist: {response}"
                    documents.append(conversation_text)
                    metadatas.append(
                        {
                            "source": dataset_name,
                            "type": "therapy_conversation",
                            "client_message": context,
                            "therapist_response": response,
                        }
                    )
                elif context:
                    # If only context available, still useful for understanding client concerns
                    documents.append(f"Client concern: {context}")
                    metadatas.append(
                        {
                            "source": dataset_name,
                            "type": "client_concern",
                            "content": context,
                        }
                    )
                elif response:
                    # If only response available, useful for therapeutic response patterns
                    documents.append(f"Therapeutic response: {response}")
                    metadatas.append(
                        {
                            "source": dataset_name,
                            "type": "therapeutic_response",
                            "content": response,
                        }
                    )

            if documents:
                self.add_documents(documents, metadatas)
                print(
                    f"Successfully loaded {len(documents)} therapy conversations from {dataset_name}"
                )
                return len(documents)
            else:
                print(f"No suitable content found in dataset {dataset_name}")
                return 0

        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {str(e)}")
            return 0

    def load_therapy_dataset(self, dataset_name: str, limit: Optional[int] = 100):
        """Load therapy-related dataset from HuggingFace"""
        try:
            print(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split="train")

            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))

            documents = []
            metadatas = []

            for item in dataset:
                # Try common text fields
                text_fields = [
                    "text",
                    "content",
                    "question",
                    "answer",
                    "dialogue",
                    "conversation",
                ]
                text_content = None

                for field in text_fields:
                    if field in item and item[field]:
                        text_content = str(item[field])
                        break

                if text_content:
                    documents.append(text_content)
                    metadatas.append(
                        {"source": dataset_name, "type": "therapy_dataset"}
                    )

            if documents:
                self.add_documents(documents, metadatas)
                print(
                    f"Successfully loaded {len(documents)} documents from {dataset_name}"
                )
            else:
                print(f"No suitable text content found in dataset {dataset_name}")

        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {str(e)}")

    def get_retriever(self, k: int = 4) -> BaseRetriever:
        """Get retriever for RAG chain"""
        return self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

    def retrieve_context(self, query: str, k: int = 4) -> List[str]:
        """Retrieve relevant context for a query"""
        retriever = self.get_retriever(k=k)
        docs = retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]

    def get_qa_chain(self):
        """Get RetrievalQA chain for question answering"""
        retriever = self.get_retriever()
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )


# Recommended Datasets for CBT/Therapy RAG:

"""
RECOMMENDED HUGGINGFACE DATASETS FOR THERAPY RAG:

1. "Amod/mental_health_counseling_conversations" 
   - Contains therapy conversations and counseling dialogues
   - Good for understanding therapeutic communication patterns

2. "heliosbrahma/mental_health_chatbot_dataset"
   - Mental health Q&A pairs
   - Useful for understanding common mental health questions and responses

3. "alexandreteles/mental-health-conversational-data"
   - Conversational data focused on mental health support
   - Contains empathetic responses and therapeutic guidance

4. "Amod/mental_health_counseling_conversations"
   - Professional counseling conversation examples
   - Demonstrates proper therapeutic communication

5. "cogito/cbt_therapy_dataset" (if available)
   - Specific CBT techniques and applications
   - Structured therapeutic interventions

To use these datasets, call:
rag_engine.load_therapy_dataset("dataset_name", limit=500)
"""

# Example usage:
if __name__ == "__main__":
    # Initialize RAG engine
    rag = RAGEngine()

    # Add CBT knowledge base
    rag.add_cbt_knowledge_base()

    # Load your preferred dataset
    print("Loading mental health counseling conversations...")
    count = rag.load_mental_health_conversations(limit=100)
    print(f"Loaded {count} therapy conversations")

    # Test retrieval with therapy-specific queries
    test_queries = [
        "I'm feeling anxious about social situations",
        "I have negative thoughts about myself",
        "I feel depressed and unmotivated",
    ]

    print("\nTesting retrieval:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        context = rag.retrieve_context(query, k=2)
        for i, ctx in enumerate(context):
            print(f"{i+1}. {ctx[:200]}...")
            if "Client:" in ctx and "Therapist:" in ctx:
                print("   → Found therapy conversation example")
