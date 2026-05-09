#!/bin/bash
set -e

cd /home/saurabh/repos/smart-city-traffic-ai

echo "2. Converting OSM to SUMO network (adding traffic lights)..."
cd sumo
netconvert --osm-files dhanbad.osm -o dhanbad.net.xml \
    --geometry.remove --roundabouts.guess --ramps.guess \
    --junctions.join --tls.guess-signals --tls.discard-simple --tls.join

echo "3. Generating Vehicle Routes..."
# Find randomTrips.py
if [ -n "$SUMO_HOME" ] && [ -f "$SUMO_HOME/tools/randomTrips.py" ]; then
    RANDOM_TRIPS="$SUMO_HOME/tools/randomTrips.py"
elif [ -f "/usr/share/sumo/tools/randomTrips.py" ]; then
    RANDOM_TRIPS="/usr/share/sumo/tools/randomTrips.py"
else
    RANDOM_TRIPS=$(find / -name "randomTrips.py" 2>/dev/null | head -n 1)
fi

echo "Using randomTrips.py at: $RANDOM_TRIPS"
python3 "$RANDOM_TRIPS" -n dhanbad.net.xml -o routes.rou.xml --end 20000 --fringe-factor 10 --period 1

echo "SUMO setup complete!"
