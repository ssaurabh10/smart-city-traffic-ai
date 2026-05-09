/**
 * KPIPanel — live backend metrics only.
 */

function KPICard({ label, value, unit, color, trend, help }) {
  return (
    <div className="kpi-item">
      <div className="kpi-value" style={{ color }}>{value}</div>
      {unit && <div className="kpi-label">{unit}</div>}
      <div className="kpi-label">{label}</div>
      {help && <div className="kpi-help">{help}</div>}
      {trend !== undefined && (
        <div className={`kpi-trend ${trend < 0 ? 'trend-down' : trend > 0 ? 'trend-up' : 'trend-flat'}`}>
          {trend > 0 ? '▲' : trend < 0 ? '▼' : '—'} {Math.abs(trend).toFixed(1)}
        </div>
      )}
    </div>
  )
}

export default function KPIPanel({ summary, vehicleCounts }) {
  const vehicles = summary?.total_vehicles ?? 0
  const queue    = summary?.total_queue    ?? 0
  const wait     = +(summary?.total_waiting ?? 0).toFixed(1)
  const trend    = summary?.queue_trend    ?? 0
  const fps      = summary?.fps            ?? 0
  const co2Rate = +(summary?.co2_rate_g_s ?? 0).toFixed(1)
  const co2Reduction = +(summary?.co2_reduction_pct ?? 0).toFixed(1)

  return (
    <div className="card">
      <div className="card-title"><span className="dot" />Current Traffic</div>
      <div className="kpi-grid">
        <KPICard label="Vehicles now" value={vehicles} color="var(--accent-blue)" help="cars currently in the simulation" trend={trend}/>
        <KPICard label="Stopped vehicles" value={queue} color="var(--accent-orange)" unit="queue" help="lower is better" />
        <KPICard label="Waiting time" value={wait} color="var(--accent-yellow)" unit="seconds" help="average delay" />
        <KPICard label="Pollution rate" value={co2Rate} color="var(--accent-purple)" unit="CO2 g/s" help="estimated emissions" />
        <KPICard label="CO2 reduction" value={`${co2Reduction > 0 ? '+' : ''}${co2Reduction}`} color={co2Reduction >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'} unit="%" help="vs fixed-timer signals" />
      </div>

      <div className="flex items-center gap-2 mt-2">
        <span className="text-xs" style={{color:'var(--text-muted)'}}>Stream</span>
        <span className="text-mono text-xs text-blue">{fps} fps</span>
        <span className="text-xs" style={{color:'var(--text-muted)'}}>updates per second</span>
      </div>
    </div>
  )
}
