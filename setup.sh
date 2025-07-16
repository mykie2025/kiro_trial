#!/bin/bash

# Setup script for context persistence solutions project

echo "Setting up Context Persistence Solutions project..."

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run configuration test
echo "Testing configuration..."
python -m pytest test/test_config.py -v

echo "Setup complete! Virtual environment is activated."
echo "To activate the virtual environment manually, run: source .venv/bin/activate"