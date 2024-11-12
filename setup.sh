#!/bin/bash

# Check if Python 3 is installed
if ! command -v python &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "kg_backend/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv kg_backend/venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source kg_backend/venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip 

# Check if requirements.txt exists
if [ ! -f "kg_backend/requirements.txt" ]; then
    echo "requirements.txt not found in kg_backend directory"
    exit 1
fi

# Install backend requirements
echo "Installing Python dependencies..."
pip install -r kg_backend/requirements.txt 

# Navigate to kg_frontend and install frontend dependencies
cd kg_frontend || { echo "kg_frontend directory not found"; exit 1; }
echo "Installing frontend dependencies..."
npm install

# Return to the root directory
cd ..

echo "Setup complete. You can now run your application."