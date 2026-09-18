#!/usr/bin/env bash
set -e

echo "=== Starting Voice Assistant Docker Services ==="

# 1. Postgres with pgvector
if [ "$(docker ps -aq -f name=voice_assistant_postgres)" ]; then
    echo "Starting existing voice_assistant_postgres container..."
    docker start voice_assistant_postgres
else
    echo "Creating and starting voice_assistant_postgres container..."
    docker run -d \
        --name voice_assistant_postgres \
        --restart always \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=voice_assistant \
        -p 5433:5432 \
        -v voice_assistant_postgres_data:/var/lib/postgresql/data \
        pgvector/pgvector:pg16
fi

# 2. Redis
if [ "$(docker ps -aq -f name=voice_assistant_redis)" ]; then
    echo "Starting existing voice_assistant_redis container..."
    docker start voice_assistant_redis
else
    echo "Creating and starting voice_assistant_redis container..."
    docker run -d \
        --name voice_assistant_redis \
        --restart always \
        -p 6379:6379 \
        -v voice_assistant_redis_data:/data \
        redis:7-alpine
fi

# 3. Ollama
if [ "$(docker ps -aq -f name=voice_assistant_ollama)" ]; then
    echo "Starting existing voice_assistant_ollama container..."
    docker start voice_assistant_ollama
else
    echo "Creating and starting voice_assistant_ollama container..."
    docker run -d \
        --name voice_assistant_ollama \
        --restart always \
        -p 11434:11434 \
        -v voice_assistant_ollama_data:/root/.ollama \
        ollama/ollama:latest
fi

echo "=== Services Started Successfully ==="
docker ps -f name=voice_assistant
