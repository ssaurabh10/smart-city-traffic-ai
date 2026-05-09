/**
 * AlertPanel — live congestion alert feed
 */
import { useState } from 'react'

const SEVERITY_COLOR = {
  critical: 'var(--accent-red)',
  warning:  'var(--accent-orange)',
  info:     'var(--accent-blue)',
}

export default function AlertPanel({ alerts }) {
  const [dismissed, setDismiss] = useState(new Set())
  const active = (alerts?.active_alerts ?? []).filter(a => !dismissed.has(a.alert_id))

  return (
    <div className="card">
      <div className="card-title">
        <span className="dot" style={{ background: 'var(--accent-red)' }} />
        Problems To Watch
        {active.length > 0 && (
          <span className="badge badge-stopped" style={{ marginLeft: 'auto' }}>
            {active.length}
          </span>
        )}
      </div>

      {active.length === 0 ? (
        <div className="no-data">
          <strong>No traffic problems right now.</strong>
          <span>The AI has not detected congestion or emergency priority events.</span>
        </div>
      ) : (
        <div className="alert-list">
          {active.map(a => (
            <div key={a.alert_id} className={`alert-item ${a.severity}`}>
              <div className="alert-head">
                <span className="alert-tls" style={{ color: SEVERITY_COLOR[a.severity] }}>
                  {a.severity === 'critical' ? 'Critical' : 'Warning'}: {a.tls_id?.slice(0, 16)}
                </span>
                <button
                  onClick={() => setDismiss(p => new Set([...p, a.alert_id]))}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '10px' }}
                >✕</button>
              </div>
              <div className="alert-msg">{a.message}</div>
              <div className="alert-time text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                Stopped vehicles: {a.queue} | Waiting: {a.wait}s
              </div>
            </div>
          ))}
        </div>
      )}

      {alerts?.new_alerts?.length > 0 && (
        <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          +{alerts.new_alerts.length} new update
        </div>
      )}
    </div>
  )
}
