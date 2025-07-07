# src/config.py
import os

# --- Paths ---
# Path to the original, large raw dataset
RAW_DATA_PATH = "data/complaints.csv"
# Path where the cleaned, smaller dataset will be saved
FILTERED_DATA_PATH = "data/filtered_complaints.csv"
# Path for the ChromaDB vector store
VECTOR_STORE_PATH = "vector_store"

# --- Models ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

# --- Text Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- RAG Prompt Template ---
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to provide concise answers based only on the following customer complaint excerpts.
Do not use any external knowledge. If the context does not contain the answer, state that you don't have enough information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""