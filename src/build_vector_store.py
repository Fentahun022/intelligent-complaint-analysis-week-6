# src/build_vector_store.py
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
# This import has been updated to fix the deprecation warning
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import FILTERED_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, VECTOR_STORE_PATH
import os
import pickle

def create_and_persist_vector_store():
    if not os.path.exists(FILTERED_DATA_PATH):
        print(f"Error: Cleaned data file not found at '{FILTERED_DATA_PATH}'.")
        print("Please run 'python -m src.preprocess_data' first.")
        return

    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        print(f"Vector store already exists at {VECTOR_STORE_PATH}. Skipping creation.")
        return

    print("--- Starting Vector Store and Document Creation ---")
    df = pd.read_csv(FILTERED_DATA_PATH)
    df.rename(columns={'cleaned_narrative': 'text'}, inplace=True)
    loader = DataFrameLoader(df, page_content_column='text')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    
    docs_pickle_path = os.path.join(VECTOR_STORE_PATH, "docs.pkl")
    # Ensure the directory exists before saving the pickle file
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    with open(docs_pickle_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} raw document chunks to {docs_pickle_path}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"Creating and persisting Chroma vector store at {VECTOR_STORE_PATH}...")
    Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_STORE_PATH)
    print("--- Vector Store Creation Complete ---")

if __name__ == "__main__":
    create_and_persist_vector_store()