#!/bin/bash
set -e

cd /home/saurabh/repos/smart-city-traffic-ai

echo "1. Downloading OSM Data for Dhanbad..."
source venv/bin/activate
python scripts/download_dhanbad.py

echo "2. Converting OSM to SUMO network (adding traffic lights)..."
cd sumo
# netconvert configuration with traffic light guessing and junction joining
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
    # Fallback to searching
    RANDOM_TRIPS=$(find / -name "randomTrips.py" 2>/dev/null | head -n 1)
fi

echo "Using randomTrips.py at: $RANDOM_TRIPS"
python3 "$RANDOM_TRIPS" -n dhanbad.net.xml -o routes.rou.xml --end 3600 --fringe-factor 10 --period 2

echo "4. Creating SUMO configuration file..."
cat <<EOF > dhanbad.sumocfg
<configuration>
    <input>
        <net-file value="dhanbad.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
EOF

echo "SUMO setup complete. You can test it by running:"
echo "sumo-gui -c sumo/dhanbad.sumocfg"
