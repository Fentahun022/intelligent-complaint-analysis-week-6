# src/rag_pipeline.py

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import VECTOR_STORE_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, PROMPT_TEMPLATE

# --- LangChain Core Imports ---
# These are the building blocks for creating a RAG chain.
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# --- LangChain Community Integrations ---
# These are the specific implementations we will use.
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma

class RAGPipeline:
    """
    A class that encapsulates the entire Retrieval-Augmented Generation pipeline
    using LangChain Expression Language (LCEL) for a more structured approach.
    """
    def __init__(self):
        """
        Initializes the RAG pipeline by setting up the retriever, the LLM, and the RAG chain.
        """
        print("--- Initializing RAG Pipeline (LangChain-Idiomatic) ---")

        # 1. Initialize the Retriever (Same as before)
        print("Loading vector store and retriever...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print("Retriever loaded successfully.")

        # 2. Initialize the Hugging Face LLM Pipeline
        # This is the same logic as before, but we will wrap it in a LangChain component.
        print(f"Loading Language Model: {LLM_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=350
        )
    
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print("LLM loaded and wrapped successfully.")
        print("Building RAG chain...")
        
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        def format_docs(docs):
            return "\n\n---\n\n".join([d.page_content for d in docs])
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("--- RAG Pipeline Initialized ---")

    def answer_question(self, question: str):
        """
        Answers a question using the pre-defined RAG chain.
        This function is now much simpler.
        """
        # We need to get the sources separately for our UI
        retrieved_docs = self.retriever.get_relevant_documents(question)
        
        # Invoke the chain to get the final answer.
        # The chain handles passing the context and question to the prompt and LLM.
        answer = self.rag_chain.invoke(question)
        
        # Return the generated answer and the retrieved source documents
        return answer, retrieved_docs