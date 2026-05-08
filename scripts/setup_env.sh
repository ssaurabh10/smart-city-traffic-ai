#!/bin/bash
set -e

echo "Setting up Python virtual environment..."
cd /home/saurabh/repos/smart-city-traffic-ai
python3 -m venv venv
source venv/bin/activate

echo "Installing Python Libraries..."
pip install --upgrade pip
pip install fastapi uvicorn websockets redis celery
pip install stable-baselines3 torch gymnasium
pip install sumo-rl traci
pip install osmnx networkx
pip install psycopg2-binary pandas numpy matplotlib

echo "Setting up Frontend..."
cd /home/saurabh/repos/smart-city-traffic-ai/frontend
if [ ! -f package.json ]; then
    npm init -y
fi

echo "Installing Frontend Libraries..."
npm install react react-dom vite tailwindcss
npm install mapbox-gl deck.gl socket.io-client recharts

echo "Environment setup complete!"
