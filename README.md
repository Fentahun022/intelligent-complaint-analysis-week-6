# Intelligent Complaint Analysis for CrediTrust Financial
# Interim Submission: Tasks 1 & 2
This document outlines the progress and key deliverables for the first phase of the "Intelligent Complaint Analysis" project. The work completed covers Task 1 (EDA and Data Preprocessing) and Task 2 (Text Chunking, Embedding, and Vector Store Indexing), laying the foundation for the full RAG-powered chatbot.
Business Objective
The goal of this project is to develop an internal AI tool that transforms raw, unstructured customer complaint data into a strategic asset. This tool will empower non-technical teams at CrediTrust Financial to query a large volume of complaints in plain English and receive synthesized, evidence-backed answers in seconds, shifting the company from a reactive to a proactive problem-solving model.

# Key Features
Analyzes Real-World Data: Ingests and processes the CFPB's public complaint dataset.
RAG Pipeline: Uses a Retrieval-Augmented Generation (RAG) architecture to find relevant complaint narratives and generate accurate answers.
Source-Backed Answers: Each answer is accompanied by snippets from the source complaints, ensuring trust and verifiability.
Interactive UI: A simple and intuitive web interface built with Gradio for easy querying by non-technical users.
# Tech Stack
Backend: Python
AI/ML Framework: LangChain, Hugging Face Transformers
Data Processing: Polars (for high-performance initial processing)
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
LLM: google/flan-t5-base
Vector Database: ChromaDB
Frontend: Gradio
 # Steps
 git clone  https://github.com/Fentahun022/intelligent-complaint-analysis-week-6.git
cd intelligent-complaint-analysis
# Create and activate a virtual environment: Using venv:

# For macOS/Linux
python3 -m venv venv source venv/bin/activate

# Install dependencies:

pip install -r requirements.txt