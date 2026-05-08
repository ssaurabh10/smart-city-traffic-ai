import requests
import sys

print("Downloading OSM data for Dhanbad from Overpass API...")

overpass_url = "http://overpass-api.de/api/interpreter"
# Bounding box for central Dhanbad (approx)
# format: (south, west, north, east)
overpass_query = """
[out:xml][timeout:250];
(
  node(23.75,86.38,23.85,86.48);
  way(23.75,86.38,23.85,86.48);
  relation(23.75,86.38,23.85,86.48);
);
out body;
>;
out skel qt;
"""

response = requests.post(overpass_url, data={'data': overpass_query})

if response.status_code == 200:
    with open("sumo/dhanbad.osm", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Successfully downloaded dhanbad.osm")
else:
    print(f"Failed to download. Status code: {response.status_code}")
    sys.exit(1)
