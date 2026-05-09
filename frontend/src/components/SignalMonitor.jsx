/**
 * SignalMonitor — live traffic light states for all TLS junctions
 */

function getDisplaySignal(sig, index) {
  const state = String(sig?.phase_state ?? '').toLowerCase()
  if (state.includes('y')) return 'yellow'
  if (state.includes('g')) return 'green'
  return 'red'
}

function SignalLights({ displaySignal }) {
  return (
    <div className="signal-lights">
      <div className={`signal-light ${displaySignal === 'red' ? 'on-red' : ''}`}
           style={{ background: 'var(--accent-red)', opacity: displaySignal === 'red' ? 1 : 0.15 }} />
      <div className={`signal-light ${displaySignal === 'yellow' ? 'on-yellow' : ''}`}
           style={{ background: 'var(--accent-yellow)', opacity: displaySignal === 'yellow' ? 1 : 0.15 }} />
      <div className={`signal-light ${displaySignal === 'green' ? 'on-green' : ''}`}
           style={{ background: 'var(--accent-green)', opacity: displaySignal === 'green' ? 1 : 0.15 }} />
    </div>
  )
}

export default function SignalMonitor({ signals, alerts, actions, overrideSignal }) {
  const activeAlerts = alerts?.active_alerts ?? []
  const criticalIds  = new Set(activeAlerts.filter(a => a.severity === 'critical').map(a => a.tls_id))
  const warnIds      = new Set(activeAlerts.filter(a => a.severity === 'warning').map(a => a.tls_id))

  const entries = Object.entries(signals ?? {})
  const tspActive = actions?.bus_priority_active === true
  const emergencyActive = actions?.emergency_active === true
  const emergencyTlsCount = actions?.emergency?.corridor_tls?.length ?? 0
  const greenWaveActive = actions?.green_wave_active === true
  const greenWaveTlsCount = actions?.green_wave?.tls_ids?.length ?? 0

  return (
    <div className="card">
      <div className="card-title">
        <span className="dot" style={{ background: 'var(--accent-green)' }} />
        Traffic Lights
        <span className="badge badge-info" style={{ marginLeft: 'auto', padding: '2px 6px' }}>
          {entries.length} signals
        </span>
      </div>

      {entries.length === 0 ? (
        <div className="no-data">
          <strong>No live signal data.</strong>
          <span>Press Start to run the backend simulation.</span>
        </div>
      ) : (
        <div className="signal-grid">
          {entries.map(([tid, sig], index) => {
            const isCritical = criticalIds.has(tid)
            const isWarn     = warnIds.has(tid)
            const displaySignal = getDisplaySignal(sig, index)
            return (
              <div
                key={tid}
                className={`signal-item ${isCritical ? 'critical' : displaySignal === 'green' ? 'active' : ''}`}
              >
                <div className="signal-id" title={tid}>
                  Junction {index + 1}
                </div>
                <SignalLights displaySignal={displaySignal} />
                <div className="signal-queue" style={{
                  color: isCritical ? 'var(--accent-red)' : isWarn ? 'var(--accent-orange)' : 'var(--text-primary)'
                }}>
                  {displaySignal === 'green' ? 'Go' : displaySignal === 'yellow' ? 'Ready' : 'Stop'}
                </div>
                {isCritical && (
                  <span className="badge badge-stopped" style={{ fontSize: '9px', padding: '1px 5px' }}>Problem</span>
                )}
                <div className="flex gap-1">
                  <button className="btn btn-ghost btn-sm" onClick={() => overrideSignal && overrideSignal(tid, 1)} style={{ padding: '2px 4px', fontSize: '9px' }}>Next</button>
                  <button className="btn btn-danger btn-sm" onClick={() => overrideSignal && overrideSignal(tid, 3)} style={{ padding: '2px 4px', fontSize: '9px' }}>Red</button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Emergency / Bus priority status */}
      <div className="flex gap-2 mt-2">
        <div className={`badge ${tspActive ? 'badge-live' : 'badge-warn'}`} style={{ opacity: tspActive ? 1 : 0.4 }}>
          {tspActive && <span className="badge-dot" />} Bus priority
        </div>
        <div className={`badge ${emergencyActive ? 'badge-live' : 'badge-stopped'}`} style={{ opacity: emergencyActive ? 1 : 0.4 }}>
          {emergencyActive && <span className="badge-dot" />} Emergency route{emergencyActive && emergencyTlsCount ? ` | ${emergencyTlsCount} signals` : ''}
        </div>
        <div className={`badge ${greenWaveActive ? 'badge-info' : 'badge-warn'}`} style={{ opacity: greenWaveActive ? 1 : 0.4 }}>
          {greenWaveActive && <span className="badge-dot" />} Green wave{greenWaveActive && greenWaveTlsCount ? ` | ${greenWaveTlsCount} signals` : ''}
        </div>
      </div>
    </div>
  )
}
