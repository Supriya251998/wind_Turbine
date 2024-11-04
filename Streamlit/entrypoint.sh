#!/bin/bash

# Start Ollama in the background.
# Ensure the models will be stored in /app/ollama, not /root/.ollama
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

# Set the directory for the model (no longer hidden)
MODEL_DIR="/root/.ollama/models/manifests"

# Check if the llama3 model is already present
if [ -d "$MODEL_DIR" ]; then
  echo "🟢 llama3 model is already pulled."
else
  echo "🔴 Retrieving llama3 model..."
  ollama pull llama3
  if [ $? -ne 0 ]; then
    echo "🔴 Failed to pull llama3 model!"
    exit 1
  fi
  echo "🟢 llama3 model pulled successfully!"
fi

# Wait for Ollama process to finish.
wait $pid
