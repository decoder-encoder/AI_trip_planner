import os
from typing import TypedDict, Annotated
import operator
import json
import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv

from tools.rag_tool import retrieve_docs
load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    streaming=True
)

# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    rewritten_query: str
    flight_results: str
    flight_source: str
    flight_destination: str 
    waiting_for_route: bool
    hotel_results: str
    itinerary: str
    llm_calls: int
    rag_enabled: bool 
    rag_response: str
    rag_context: str 
    route: str  

#rewritingquery agent
def query_rewrite_agent(state: TravelState):

    query = state["user_query"]

    prompt = f"""
    You are a query rewriting assistant.

    Rewrite the user's travel query to make it:
    - more specific
    - better for web search
    - better for RAG retrieval
    - clearer and more detailed

    User Query:
    {query}

    Return ONLY the rewritten query.
    """

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    rewritten_query = response.content.strip()

    return {
        "rewritten_query": rewritten_query,
        "messages": [
            AIMessage(
                content=f"Query rewritten: {rewritten_query}"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
#router agent
def router_agent(state: TravelState):
    if not state.get("rag_enabled", False):
        return {
            "route": "web"
        }

    query = state.get("rewritten_query") or state["user_query"]

    prompt = f"""
You are a routing agent.

Decide which source should answer the query.

Options:

rag
- when the answer is likely present in uploaded PDFs

web
- when current internet information is needed

hybrid
- when both document knowledge and web search are useful

Query:
{query}

Return ONLY:
rag
web
hybrid
"""

    response = llm.invoke(prompt)

    route = response.content.strip().lower()

    if route not in ["rag", "web", "hybrid"]:
        route = "web"

    return {
        "route": route,
        "messages": [
        AIMessage(content=f"Routing decision: {route}")
    ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
def route_query(state: TravelState):
    return state["route"]
#web agent
def web_agent(state: TravelState):

    query = state["rewritten_query"]

    web_results = tavily_search(query)

    return {
        "rag_context": web_results,
        "messages": [
            AIMessage(content="Web search completed")
        ]
    }
#hybrid agent
def hybrid_agent(state: TravelState):

    query = state["rewritten_query"]

    rag_context = retrieve_docs(query)

    web_context = tavily_search(query)

    merged_context = f"""
RAG RESULTS:
{rag_context}

WEB RESULTS:
{web_context}
"""

    return {
        "rag_context": merged_context,
        "messages": [
            AIMessage(content="Hybrid retrieval completed")
        ]
    }


def is_relevant(context: str, query: str) -> bool:
    prompt = f"""
You are a relevance checker.

Task: Decide if the context is useful to answer the query.

Return ONLY:
YES or NO

Query:
{query}

Context:
{context}
"""

    result = llm.invoke(prompt).content.strip().lower()

    return "yes" in result
#Rag agent
def rag_agent(state):

    # 🔥 GATE (IMPORTANT)
    if not state.get("rag_enabled", False):
        return {
            "rag_response": "",
            "rag_context": "",
            "messages":   [
                AIMessage(content="RAG skipped (not enabled)")
            ]
            
        }

    query = state["rewritten_query"] or state["user_query"]

    context = retrieve_docs(query)

    if not is_relevant(context, query):
        return {
            "rag_response": "No relevant information found in documents.",
            "rag_context": context,
            "messages":  [
                AIMessage(content="RAG skipped: irrelevant context")
            ],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    prompt = f"""
You are a strict RAG assistant.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return {
        "rag_response": response.content,
        "rag_context": context,
        "messages": [
            AIMessage(content=response.content)
        ],
        "llm_calls": state.get("llm_calls", 0) + 2
    }




# Flight Agent

def flight_agent(state: TravelState):

    query = state["rewritten_query"]

    prompt = f"""
Extract the source city and destination city from the travel query.

Query:
{query}

Return ONLY valid JSON.

Example:
{{
    "source": "Delhi",
    "destination": "Mumbai"
}}

If not found:

{{
    "source": "",
    "destination": ""
}}
"""

    try:

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        data = json.loads(
            response.content.strip()
        )

        source = data.get(
            "source",
            ""
        ).strip()

        destination = data.get(
            "destination",
            ""
        ).strip()

    except Exception:

        source = ""
        destination = ""

    print("SOURCE =", source)
    print("DESTINATION =", destination)

    flight_data = search_flights(
        source,
        destination
    )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight results fetched"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 2
    }

# Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['rewritten_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

#Itinerary Agent
def itinerary_agent(state: TravelState):

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['rewritten_query']}

    Retrieved Context:
    {state['rag_context']}


    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Final Response Agent
def final_agent(state: TravelState):

    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    response = llm.invoke(final_prompt)

    return {
        "messages": [AIMessage(content=response.content)],
        "itinerary": response.content  
    }

graph = StateGraph(TravelState)

graph.add_node("rewritten_query", query_rewrite_agent)
graph.add_node("router_agent", router_agent)
graph.add_node("rag_agent", rag_agent)
graph.add_node("web_agent", web_agent)
graph.add_node("hybrid_agent", hybrid_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "rewritten_query")
graph.add_edge("rewritten_query", "router_agent")
graph.add_conditional_edges("router_agent",
    route_query,
    {
        "rag": "rag_agent",
        "web": "web_agent",
        "hybrid": "hybrid_agent"
    }
)
graph.add_edge("rag_agent","flight_agent")
graph.add_edge("web_agent", "flight_agent")
graph.add_edge("hybrid_agent", "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# Persistent connection so both CLI and Streamlit can share the compiled app
from psycopg_pool import ConnectionPool
if "sslmode" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require"

pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=5,
    open=True,
    timeout=30,
    max_lifetime=600
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)
chatbot = app

if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")

    # STEP 1: Build RAG knowledge base
    from tools.rag_tool import build_vectorstore_from_pdf

    build_vectorstore_from_pdf(pdf_path)

    print("PDF indexed successfully!\n")
    thread_id = input("Enter thread id: ")

    config = {
        "configurable": {
        "thread_id": thread_id
    }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "rewritten_query":"",
            "flight_results": "",
            "flight_source": "",
            "flight_destination": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
            "route": "",
            "rag_enabled": True,
            "rag_response": "",
            "rag_context": ""
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)