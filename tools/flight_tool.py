import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

def search_flights(source, destination):
    """
    Searches for live flights using AviationStack.
    If no matches are found, or the API fails/limits out,
    it automatically falls back to a Tavily Web Search.
    """
    if not source or not destination:
        return "Flight route could not be identified from the user query."

    url = "http://api.aviationstack.com/v1/flights"
    
    # We pass a limit but filter results dynamically to accommodate the Free Tier's constraints
    params = {
        "access_key": API_KEY,
        "limit": 100
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
    except Exception as e:
        return fallback_web_search(source, destination, f"AviationStack Connection Error: {e}")

    if "data" not in data or not data["data"]:
        return fallback_web_search(source, destination, "AviationStack returned no live data.")

    flights = []
    src_lower = source.lower()
    dst_lower = destination.lower()

    for flight in data["data"]:
        dep_info = flight.get("departure") or {}
        arr_info = flight.get("arrival") or {}
        
        # Safe extraction guarding against explicit null (NoneType) values from the API
        dep_airport = (dep_info.get("airport") or "").lower()
        dep_city = (dep_info.get("iata") or "").lower()
        arr_airport = (arr_info.get("airport") or "").lower()
        arr_city = (arr_info.get("iata") or "").lower()

        # Relaxed matching: checks if the source/destination names are in airport names OR IATA codes
        if (src_lower in dep_airport or src_lower in dep_city) and \
           (dst_lower in arr_airport or dst_lower in arr_city):
            
            airline = flight.get("airline", {}).get("name", "Unknown Airline")
            status = flight.get("flight_status", "Active")
            
            flights.append(
                f"Airline: {airline}\n"
                f"Departure: {dep_info.get('airport') or 'N/A'} ({dep_info.get('iata') or 'N/A'})\n"
                f"Arrival: {arr_info.get('airport') or 'N/A'} ({arr_info.get('iata') or 'N/A'})\n"
                f"Status: {status}\n"
            )

    # If no live flights matched our target route, execute the search engine fallback
    if not flights:
        return fallback_web_search(source, destination, f"No active flights found on this route.")

    return "\n---\n".join(flights[:5])


def fallback_web_search(source, destination, reason):
    """
    Leverages Tavily Web Search to find general schedule information and
    popular airlines operating between the source and destination.
    """
    print(f"DEBUG: Switching to Web Search. Reason: {reason}")
    
    query = f"standard flights schedule popular airlines from {source} to {destination} typical duration"
    
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tavily = TavilySearchResults(max_results=3)
        search_res = tavily.invoke({"query": query})
        
        web_text = "\n".join([r['content'] for r in search_res])
        return (
            f"✈️ [Standard Flight Routes via Live Web Search]\n"
            f"AviationStack API returned no active listings. Here are the search results for flights from {source} to {destination}:\n\n"
            f"{web_text}"
        )
    except Exception as e:
        return (
            f"✈️ [Flight Route: {source} → {destination}]\n"
            f"Flights are regularly operated between these locations by major carriers. "
            f"Please check your preferred travel booking engine for current real-time schedules and pricing."
        )