# src/build_vector_store.py
import os
import sys

# --- Path Modification ---
# This MUST be at the top of the file, before any local project imports.
# It adds the project's root directory (the parent of 'src') to Python's path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Now, we can safely import from our project ---
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import (
    FILTERED_DATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_PATH
)

def create_and_persist_vector_store():
    """
    Loads the pre-cleaned data, chunks it, creates embeddings,
    and persists the vector store for the RAG pipeline.
    """
    if not os.path.exists(FILTERED_DATA_PATH):
        print(f"Error: Cleaned data file not found at '{FILTERED_DATA_PATH}'.")
        print("Please run the preprocessing script first: `python src/preprocess_data.py`")
        return

    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
         print(f"Vector store already exists at {VECTOR_STORE_PATH}. Skipping creation.")
         return

    print("--- Starting Vector Store Creation ---")
    
    df = pd.read_csv(FILTERED_DATA_PATH)
    df.dropna(subset=['cleaned_narrative'], inplace=True)
    df.rename(columns={'cleaned_narrative': 'text'}, inplace=True)
    
    loader = DataFrameLoader(df, page_content_column='text')
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} text chunks.")
    
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"Creating and persisting vector store at {VECTOR_STORE_PATH}...")
    Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_STORE_PATH)
    print("--- ✅ Vector Store Creation Complete ---")

# This block ensures the script can be run directly using "python src/build_vector_store.py"
if __name__ == "__main__":
    create_and_persist_vector_store()