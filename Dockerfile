Lightweight Python image

FROM python:3.9-slim

Install essential system dependencies

RUN apt-get update && apt-get install -y --no-install-recommends 

build-essential 

curl 

software-properties-common 

&& rm -rf /var/lib/apt/lists/*

Set the working directory inside the container

WORKDIR /app

Copy requirements file first to leverage Docker's caching mechanism

COPY requirements.txt .

Install Python dependencies

RUN pip install --no-cache-dir -r requirements.txt

Copy the rest of your application code into the container

COPY . .

Expose the default Streamlit port

EXPOSE 8501

Healthcheck to ensure the container is running properly

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

Command to run the application

# AI_trip_planner/Dockerfile mein ye line change karo:
ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]