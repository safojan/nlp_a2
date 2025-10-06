#!/bin/bash

# Quick start script for FinTech Forecasting Application

echo "Starting FinTech Forecasting Application..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    bash setup.sh
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if required files exist
if [ ! -f "app/streamlit_app.py" ]; then
    echo "Error: streamlit_app.py not found!"
    echo "Please ensure all files are in the correct locations."
    exit 1
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Streamlit app
echo ""
echo "======================================"
echo "Launching Streamlit Application..."
echo "======================================"
echo ""
echo "The app will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0