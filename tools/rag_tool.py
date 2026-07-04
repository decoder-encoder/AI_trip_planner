
import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Embeddings configuration
# Google Gemini API key aapke environment variables mein honi chahiye
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
)

# 2. Pinecone Index Name (Pinecone console par jo naam rakha hai wahi yahan daalein)
index_name = "travel-data"

def get_vector_store(documents=None):
    """
    Pinecone Cloud RAG operation:
    1. Agar documents hain -> Chunking + Filtering (Empty chunks hatana) + Upsert
    2. Agar documents nahi hain -> Sirf Querying ke liye vectorstore return karega
    """
    if documents:
        # Step 1: Text Splitter (Empty chunks ko filter karne ke liye zaroori)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Step 2: Sirf wo chunks rakho jismein valid text hai
        final_docs = [d for d in splits if d.page_content.strip()]
        
        # Step 3: Pinecone mein upsert
        vectorstore = PineconeVectorStore.from_documents(
            final_docs, 
            embeddings, 
            index_name=index_name
        )
        return vectorstore
    
    else:
        # Querying ke liye connection
        return PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings
        )

def retrieve_docs(query: str):
    """
    Query ke liye Pinecone se relevant documents fetch karega.
    """
    try:
        db = get_vector_store()
        # Pinecone se vector search perform karein
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        return f"Error querying Pinecone index: {e}"