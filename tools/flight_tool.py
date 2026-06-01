import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv(
    "AVIATIONSTACK_API_KEY"
)


def search_flights(
    source,
    destination
):

    if not source or not destination:

        return (
            "Flight route could not be identified "
            "from the user query."
        )

    url = (
        "http://api.aviationstack.com/v1/flights"
    )

    params = {
        "access_key": API_KEY,
        "limit": 100
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        data = response.json()

    except Exception as e:

        return f"Flight API Error: {e}"

    if "data" not in data:

        return (
            f"No flight data available "
            f"for {source} → {destination}"
        )

    flights = []

    for flight in data["data"]:

        departure_airport = (
            flight.get("departure", {})
            .get("airport", "")
        )

        arrival_airport = (
            flight.get("arrival", {})
            .get("airport", "")
        )

        if (
            source.lower()
            not in departure_airport.lower()
        ):
            continue

        if (
            destination.lower()
            not in arrival_airport.lower()
        ):
            continue

        airline = (
            flight.get("airline", {})
            .get("name", "Unknown")
        )

        status = flight.get(
            "flight_status",
            "Unknown"
        )

        flights.append(
            f"""
Airline: {airline}
Departure: {departure_airport}
Arrival: {arrival_airport}
Status: {status}
"""
        )

    if not flights:

        return (
            f"No matching flights found "
            f"from {source} to {destination}."
        )

    return "\n".join(flights[:5])