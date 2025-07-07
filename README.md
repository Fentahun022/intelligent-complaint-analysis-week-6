# Intelligent Complaint Analysis for CrediTrust Financial
# Interim Submission: Tasks 1 & 2
This document outlines the progress and key deliverables for the first phase of the "Intelligent Complaint Analysis" project. The work completed covers Task 1 (EDA and Data Preprocessing) and Task 2 (Text Chunking, Embedding, and Vector Store Indexing), laying the foundation for the full RAG-powered chatbot.
Business Objective
The goal of this project is to develop an internal AI tool that transforms raw, unstructured customer complaint data into a strategic asset. This tool will empower non-technical teams at CrediTrust Financial to query a large volume of complaints in plain English and receive synthesized, evidence-backed answers in seconds, shifting the company from a reactive to a proactive problem-solving model.

# Project Status
Task 1: Exploratory Data Analysis and Data Preprocessing: Complete.
Task 2: Text Chunking, Embedding, and Vector Store Indexing: Complete.

# Task 1: Exploratory Data Analysis and Data Preprocessing
This task focused on understanding the structure of the raw CFPB complaint data and preparing it for the RAG pipeline.
Key Findings from EDA
The initial exploratory analysis was conducted using the Polars library for high-performance data manipulation on the large dataset. Key findings that directly informed our preprocessing strategy include:
Narrative Availability: Out of more than 4 million complaints, only about one-third contain a Consumer complaint narrative. This made it essential to filter out all records without this text, as it is the core input for our system.
Product Distribution: The complaint volume is heavily skewed towards a few product categories, with "Credit reporting..." being the largest. Our filtering successfully narrowed the dataset to the five relevant business areas for CrediTrust.
Narrative Length Variability: The word count of complaint narratives varies dramatically, from a few words to over 2,000. This long-tail distribution confirms that embedding entire narratives is inefficient and would lead to poor retrieval quality. This finding validates the need for a robust text chunking strategy, which was implemented in Task 2.
Deliverables
# EDA Notebook (notebooks/01_EDA_with_Polars.ipynb): A Jupyter Notebook containing the detailed analysis, visualizations, and findings.
Preprocessing Script (src/preprocess_data.py): An efficient Polars-based script that loads the raw data, applies the filtering and cleaning logic derived from the EDA, and saves a final, sampled dataset.
Cleaned Dataset (data/filtered_complaints.csv): The output of the preprocessing script. This file contains 5,000 cleaned and filtered complaint records ready for the next stage.
# Task 2: Text Chunking, Embedding, and Vector Store Indexing
This task focused on converting the cleaned text narratives into a searchable vector database.
Technical Choices and Justification
Several key technical decisions were made to build an efficient and effective retrieval backbone:
Text Chunking Strategy:
Tool: RecursiveCharacterTextSplitter from LangChain.
Parameters: chunk_size=1000, chunk_overlap=200.
Justification: This splitter is ideal as it attempts to break text along semantic boundaries (paragraphs, sentences) first. The chosen chunk size provides a good balance, capturing sufficient context within each chunk without being too large for the embedding model, while the overlap prevents losing important information at chunk boundaries.
Embedding Model:
Model: sentence-transformers/all-MiniLM-L6-v2.
Justification: This model was chosen for its excellent balance of speed, performance, and low resource requirements. It is highly effective for semantic search tasks and is ideal for running on local development machines without requiring high-end GPUs.
Vector Database:
Tool: ChromaDB.
Justification: ChromaDB is a lightweight, open-source vector database that is extremely easy to set up and persists data locally to disk. This makes it perfect for rapid development and prototyping without the overhead of setting up a cloud-based or server-based database.
Deliverables
Indexing Script (src/build_vector_store.py): A Python script that loads the cleaned data, applies the chunking strategy, generates embeddings using the chosen model, and saves the results into a persistent ChromaDB database.
Vector Store (vector_store/ directory): The persisted ChromaDB database containing the embedded text chunks and their associated metadata (Product, Complaint ID).
 # Steps
 git clone  https://github.com/Fentahun022/intelligent-complaint-analysis-week-6.git
cd intelligent-complaint-analysis
# Create and activate a virtual environment: Using venv:

# For macOS/Linux
python3 -m venv venv source venv/bin/activate

# Install dependencies:

pip install -r requirements.txt