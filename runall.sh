#!/bin/bash

echo "Running Team5.py..."
python3 Team5.py || echo "Team5.py failed."

echo "Running ai.py..."
python3 ai.py || echo "ai.py failed."

echo "Running Team5_2.0.py..."
python3 Team5_2.0.py || echo "Team5_2.0.py failed."

echo "Running gemini.py..."
python3 gemini.py || echo "gemini.py failed."

echo "All scripts attempted."
