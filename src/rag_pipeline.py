# src/rag_pipeline.py
import os
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import *

# --- LangChain Core Imports ---
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# --- LangChain Community Integrations ---
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma

class RAGPipeline:
    def __init__(self):
        print("--- Initializing OPTIMIZED RAG Pipeline (Hybrid Search) ---")

        # 1. Load Documents and Initialize Retrievers
        print("Loading documents and initializing retrievers...")
        
        # Load the raw documents for the keyword retriever
        docs_pickle_path = os.path.join(VECTOR_STORE_PATH, "docs.pkl")
        with open(docs_pickle_path, "rb") as f:
            all_splits = pickle.load(f)

        # Initialize Keyword (BM25) Retriever
        bm25_retriever = BM25Retriever.from_documents(all_splits)
        bm25_retriever.k = 5 # Retrieve top 5 keyword matches

        # Initialize Vector (Semantic) Retriever
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
        chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # --- THIS IS THE KEY OPTIMIZATION ---
        # Initialize Ensemble Retriever to combine both search methods
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5] # Give equal weight to both methods
        )
        print("Hybrid search retriever loaded successfully.")
        # --- END OF OPTIMIZATION ---

        # 2. Initialize the LLM (same as before)
        print(f"Loading Language Model: {LLM_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
        hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=350)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print("LLM loaded successfully.")

        # 3. Build the RAG Chain using LCEL (now with the ensemble retriever)
        print("Building RAG chain...")
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        
        def format_docs(docs):
            return "\n\n---\n\n".join([d.page_content for d in docs])

        self.rag_chain = (
            {"context": self.ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("--- RAG Pipeline Initialized ---")

    def answer_question(self, question: str):
        # We need the sources for the UI, so we run the retriever separately first
        retrieved_docs = self.ensemble_retriever.get_relevant_documents(question)
        answer = self.rag_chain.invoke(question)
        return answer, retrieved_docs