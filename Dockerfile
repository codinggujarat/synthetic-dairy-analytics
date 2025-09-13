# Dockerfile - builds the flask app + worker image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for some packages (e.g., build, fonts for fpdf if needed)
RUN apt-get update && apt-get install -y build-essential libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app code
COPY . /app

# Create directory for instance DB if needed
RUN mkdir -p /app/instance && chmod -R 777 /app/instance

# Expose port
EXPOSE 5000

# Default entrypoint for dev; overridden by docker-compose for worker
CMD ["flask", "run", "--host=0.0.0.0"]
