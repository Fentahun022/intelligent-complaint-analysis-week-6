# app.py

import gradio as gr
from src.rag_pipeline import RAGPipeline
import os
from src.config import VECTOR_STORE_PATH

def main():
    """
    Main function to initialize the RAG pipeline and launch the Gradio web app.
    """
    # --- Health Check: Ensure the vector store exists before starting ---
    if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
        print("="*80)
        print("ERROR: Vector store not found! The 'brain' of the chatbot is missing.")
        print(f"Please make sure the vector store exists at '{VECTOR_STORE_PATH}'.")
        print("You can create it by running the following scripts in order:")
        print("1. `python src/preprocess_data.py`")
        print("2. `python src/build_vector_store.py`")
        print("="*80)
        return

    # --- Initialization ---
    print("Initializing application... This may take a moment to load the models.")
    rag_pipeline = RAGPipeline()
    print("Application initialized successfully. Launching web interface...")

    # --- Define the Chat Interface Logic ---
    def chat_interface(question, history):
        """
        The core function that powers the Gradio chat interface.
        It takes a user's question and returns a formatted answer with sources.
        """
        answer, sources = rag_pipeline.answer_question(question)
        
        # --- Enhancing Trust and Usability: Display Sources ---
        # Format the retrieved source documents for display
        source_text = "\n\n---\n\n**Sources Used for this Answer:**\n"
        for i, doc in enumerate(sources):
            # Extract metadata and a snippet of the content
            product = doc.metadata.get('Product', 'N/A')
            content_snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            source_text += f"\n**Source {i+1}** (Product: {product})\n"
            source_text += f"> {content_snippet}\n"
        
        # Combine the generated answer with the source text
        return answer + source_text

    # --- Build and Launch the Gradio App ---
    # The modern gr.ChatInterface includes clear and undo buttons by default.
    # We only need to provide the core functionality.
    gr.ChatInterface(
        fn=chat_interface,
        title="CrediTrust Intelligent Complaint Analysis 📈",
        description="Ask questions in plain English about customer complaints to get synthesized, evidence-backed answers.",
        examples=[
            "Why are people unhappy with their personal loans?",
            "What are the main issues with money transfers?",
            "Summarize the top 3 problems related to credit cards.",
        ],
        cache_examples=False,
        theme="soft"
    ).launch()

if __name__ == "__main__":
    main()