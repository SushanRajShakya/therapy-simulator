import os
import pinecone
from datasets import load_dataset
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from server.config import *

# Set up Pinecone
pinecone.init(api_key=PINECONE_API_KEY)

# Model and Embeddings
llm = OpenAI(model=MODEL_CONFIG["model"], openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)


def setup_pinecone_index(index_name: str):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=embeddings.embedding_dimensions)
    return pinecone.Index(index_name)


class RAGEngine:
    def __init__(self, index_name: str):
        self.index = setup_pinecone_index(index_name)
        self.vectorstore = Pinecone(self.index, embeddings.embed_query, "text")
        self.llm = llm
        self.text_splitter = text_splitter

    def add_documents(self, documents):
        # documents: list of dicts with 'text' field
        texts = [doc["text"] for doc in documents]
        chunks = self.text_splitter.create_documents(texts)
        self.vectorstore.add_documents(chunks)

    def add_dataset(
        self,
        dataset_path: str,
        split: str = "train",
        text_key: str = "text",
        limit: int = None,
    ):
        """
        Loads a Hugging Face dataset and adds its text to the vectorstore.
        Args:
            dataset_path: Hugging Face dataset path (e.g. 'Psychotherapy-LLM/CBT-Bench')
            split: Dataset split (e.g. 'train')
            text_key: Key for text field in dataset
            limit: Optional limit on number of rows
        """
        dataset = load_dataset(dataset_path, split=split)
        if limit:
            dataset = dataset.select(range(limit))
        docs = []
        for item in dataset:
            if text_key in item:
                docs.append({"text": item[text_key]})
        self.add_documents(docs)

    def query(self, question: str):
        retriever = self.vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=retriever)
        return qa_chain.run(question)


# Example usage (to be integrated in FastAPI endpoints or LCEL workflows):
# rag = RAGEngine(index_name="therapy-cbt")
# rag.add_dataset("Psychotherapy-LLM/CBT-Bench", split="train", text_key="text", limit=100)
# rag.add_dataset("epsilon3/cbt-cognitive-distortions-analysis", split="train", text_key="text", limit=100)
# response = rag.query("How does CBT help with negative thoughts?")
# print(response)
