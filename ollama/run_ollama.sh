#!/bin/bash

echo "Starting Ollama server..."
ollama serve &  # Start Ollama in the background
sleep 5

echo "Ollama is ready, creating the model..."
ollama create deepseek -f model_files/Modelfile

echo "Model created, running the model...."
ollama run deepseek