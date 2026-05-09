function StatusPill({ label, value, tone = 'info' }) {
  return (
    <div className={`overview-pill overview-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export default function OverviewPanel({ status, connected, summary, alerts }) {
  const running = status?.running === true
  const aiOn = status?.ai_enabled === true
  const vehicles = summary?.total_vehicles ?? 0
  const queue = summary?.total_queue ?? 0
  const waiting = +(summary?.total_waiting ?? 0).toFixed(1)
  const activeAlerts = alerts?.active_alerts?.length ?? 0

  let headline = 'Simulation is ready'
  let detail = 'Press Start to run the Dhanbad traffic simulation.'

  if (running && vehicles === 0) {
    headline = 'Simulation is running, but traffic is currently clear'
    detail = 'The city clock is moving. No vehicles are present at this moment, so queues and wait time are zero.'
  } else if (running && queue === 0) {
    headline = 'Traffic is flowing smoothly'
    detail = `${vehicles} vehicles are moving with no stopped queue right now.`
  } else if (running) {
    headline = 'Traffic needs attention'
    detail = `${queue} vehicles are stopped and average waiting time is ${waiting} seconds.`
  }

  return (
    <div className="card overview-card">
      <div className="card-title"><span className="dot" />What am I looking at?</div>
      <h2>{headline}</h2>
      <p>{detail}</p>

      <div className="overview-grid">
        <StatusPill label="Simulation" value={running ? 'Running' : 'Stopped'} tone={running ? 'good' : 'warn'} />
        <StatusPill label="AI control" value={aiOn ? 'On' : 'Off'} tone={aiOn ? 'good' : 'warn'} />
        <StatusPill label="Live feed" value={connected ? 'Connected' : 'Disconnected'} tone={connected ? 'good' : 'bad'} />
        <StatusPill label="Alerts" value={activeAlerts} tone={activeAlerts > 0 ? 'bad' : 'good'} />
      </div>
    </div>
  )
}
