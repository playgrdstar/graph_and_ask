#!/bin/bash

# Check if Python 3 is installed
if ! command -v python &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Activate the virtual environment
if [ -d "kg_backend/venv" ]; then
    echo "Activating virtual environment..."
    source kg_backend/venv/bin/activate
else
    echo "Virtual environment not found. Please create it first."
    exit 1
fi

# Start the backend server in the background
echo "Starting the backend server..."
python kg_backend/manage.py runserver &

# Navigate to the frontend directory and start the frontend server
cd kg_frontend || { echo "kg_frontend directory not found"; exit 1; }
echo "Starting the frontend development server..."
npm start

# Wait for the backend server to finish (if it ever does)
wait