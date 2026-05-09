import React from 'react';
import Map, { Marker } from 'react-map-gl/maplibre';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

export default function LiveMap({ heatmap, signalStates }) {
  const junctions = heatmap?.junctions ?? []
  const signals = signalStates?.signals ?? {}
  const hotspots = heatmap?.pollution_hotspots ?? []
  const hotspotIds = new Set(hotspots.slice(0, 3).map(h => h.tls_id))

  // Center roughly on Dhanbad
  const initialViewState = {
    longitude: 86.43,
    latitude: 23.795,
    zoom: 12,
    pitch: 45
  };

  return (
    <div className="map-area" style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Map
        mapLib={maplibregl}
        initialViewState={initialViewState}
        style={{ width: '100%', height: '100%' }}
        mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
      >
        {junctions.map((jct) => {
          const sig = signals[jct.tls_id];
          const state = String(sig?.phase_state ?? '').toLowerCase()
          const color = state.includes('y') ? '#ffbe2e' : state.includes('g') ? '#00a91c' : '#d54309'
          
          const isHotspot = hotspotIds.has(jct.tls_id)
          const size = 12 + (jct.intensity ?? 0) * 16

          return (
            <Marker key={jct.tls_id} longitude={jct.lon} latitude={jct.lat} anchor="center">
              <div style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{
                  width: size,
                  height: size,
                  borderRadius: '50%',
                  backgroundColor: color,
                  opacity: 0.85,
                  boxShadow: `0 1px 4px rgba(0,0,0,0.3)`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  fontFamily: 'Helvetica',
                  border: isHotspot ? '2px dashed #5c2d91' : '1px solid rgba(0,0,0,0.2)',
                  transition: 'background-color 0.3s'
                }}>
                  {jct.queue > 0 ? jct.queue : ''}
                </div>
                {isHotspot && (
                  <div style={{ position: 'absolute', top: -16, background: '#5c2d91', color: 'white', fontSize: '9px', padding: '2px 4px', borderRadius: '3px', whiteSpace: 'nowrap', fontWeight: 'bold' }}>
                    High CO2
                  </div>
                )}
              </div>
            </Marker>
          )
        })}
      </Map>

      <div className="map-overlay">
        <div className="map-badge">
          <strong>{junctions.length}</strong> traffic lights monitored
        </div>
        {hotspots[0] && (
          <div className="map-badge">
            CO2 hotspot <strong>{Number(hotspots[0].co2_rate_g_s ?? 0).toFixed(1)} g/s</strong>
          </div>
        )}
        <div className="map-badge">
          Dhanbad, Jharkhand
        </div>
      </div>
    </div>
  )
}
