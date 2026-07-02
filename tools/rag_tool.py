import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# CHANGE: HuggingFace hataya, Google Gemini add kiya
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage

load_dotenv()

# .env file se apni Google API key uthao
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# CHANGE: Initialize Google Embedding model (No local Torch required)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    # task_type="retrieval_document",
    google_api_key=GOOGLE_API_KEY
)

VECTORSTORE_PATH = "faiss_index"

def build_vectorstore_from_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Ab vectorization Google Cloud par hoga, aapki RAM use nahi hogi
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return "Vectorstore created successfully using Google Gemini API!"

def load_vectorstore():
    # FAISS local index load karega, embeddings sirf query ke liye use hogi
    return FAISS.load_local(
        VECTORSTORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

def retrieve_docs(query: str):
    try:
        db = load_vectorstore()
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        return f"Error loading index: {e}"