#!/bin/bash

# Start MLflow server on port 80
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 80 &

if [ $? -ne 0 ]; then
  echo "Failed to start MLflow server"
  exit 1
fi

# Start the Streamlit app on port 8501
streamlit run ui.py --server.port 8501
