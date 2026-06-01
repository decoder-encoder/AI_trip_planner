# ✈️ WanderFlow AI – Multi-Agent Travel Planning System (LangGraph + RAG)

An intelligent AI-powered travel planning system that generates complete travel itineraries including **flights, hotels, and day-wise plans** using a multi-agent architecture built with **LangGraph, LangChain, and Groq LLMs**.

---

## Features

-  AI-powered flight search using AviationStack API  
-  Hotel recommendations using Tavily web search  
-  Multi-agent orchestration using LangGraph  
-  RAG (Retrieval-Augmented Generation) using PDF knowledge base  
-  Intelligent routing (Web / RAG / Hybrid)  
-  Persistent chat memory using PostgreSQL (Neon compatible)  
-  Modern Streamlit UI with streaming agent execution  
-  Real-time travel itinerary generation  

---

##  System Architecture

User Query  
→ Query Rewrite Agent  
→ Router Agent (RAG / Web / Hybrid)  
→ Flight Agent (route extraction + flight search)  
→ Hotel Agent (web search)  
→ Itinerary Agent (AI planning)  
→ Final Agent (response generation)

---

## Tech Stack

- Python  
- LangGraph  
- LangChain  
- Groq LLaMA 3.3 70B  
- Streamlit  
- PostgreSQL (Neon Cloud DB supported)  
- Tavily Search API  
- AviationStack API  
- FAISS (for vector storage)

---

## Project Structure
multiagentdemo/
│
├── main.py # LangGraph backend
├── frontend.py # Streamlit UI
├── tools/
│ ├── rag_tool.py
│ ├── tavily_tool.py
│ ├── flight_tool.py
│
├── .env # API keys (not pushed to GitHub)
└── README.md

## setup instructions
    1. Clone repository
        git clone https://github.com/your-username/wanderflow-ai.git
        cd wanderflow-ai

    2. Create virtual environment
        python -m venv venv
    
    3. Activate virtual environment
        Windows
            venv\Scripts\activate
        Mac/Linux
            source venv/bin/activate
    4. Install dependencies
        pip install -r requirements.txt

    5. Add environment variables
        Create a .env file in the root directory:

        GROQ_API_KEY=your_key
        AVIATIONSTACK_API_KEY=your_key
        TAVILY_API_KEY=your_key
        DATABASE_URL=your_neon_postgres_url
    6. Run backend
        python main.py
    7. Run frontend
        streamlit run frontend.py
💡 Example Usage
    Plan a 5-day trip from Delhi to Japan under 2 lakh budget including flights and hotels
##  Future Improvements

### Real-time Integrations
- Live flight booking system integration  
- Google Maps API for route visualization  

###  AI Enhancements
- AI-based price prediction for flights and hotels  
- Smarter itinerary optimization using user behavior  

###  Collaboration Features
- Multi-user trip planning  
- Shared itinerary editing (like Google Docs)  

###  Monetization Features
- Payment gateway integration for bookings  
- Premium AI travel assistant tier  

###  Product Expansion
- Mobile app version (React Native / Flutter)  
- Offline travel plan access  
##  Author

###  Pragati Varshney
-  Aspiring Software Engineer
-  Passionate about AI, Multi-Agent Systems & Full Stack Development
-  Built WanderFlow AI – Multi-Agent Travel Planner
-  Focused on LangGraph, RAG systems, and scalable backend architectures
