import { useTrafficStream }  from './hooks/useTrafficStream.js'
import { useSimControl }     from './hooks/useSimControl.js'
import KPIPanel              from './components/KPIPanel.jsx'
import SignalMonitor         from './components/SignalMonitor.jsx'
import AlertPanel            from './components/AlertPanel.jsx'
import TrafficCharts         from './components/TrafficCharts.jsx'
import LiveMap               from './components/LiveMap.jsx'
import OverviewPanel         from './components/OverviewPanel.jsx'
import { useState }          from 'react'

export default function App() {
  const { bundle, connected } = useTrafficStream()
  const { status, loading, startSim, stopSim, resetSim, toggleAI, overrideSignal } = useSimControl()
  const [aiOn, setAiOn] = useState(true)

  const summary       = bundle?.summary
  const vehicleCounts = bundle?.vehicle_counts
  const heatmap       = bundle?.heatmap
  const signalStates  = bundle?.signal_states
  const alerts        = bundle?.alerts
  const history       = summary?.history ?? []

  const isRunning = status?.running ?? false
  const step      = status?.step ?? 0

  const handleStart = () => { startSim(aiOn) }
  const handleToggleAI = () => { setAiOn(p => !p); toggleAI() }

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <div className="logo-dot" />
          <div>
            <h1>Smart City Traffic AI</h1>
            <span>Dhanbad traffic simulation</span>
          </div>
        </div>

        <div className="header-meta">
          <div className="meta-item">
            <span>Live feed</span>
            <span className={`meta-value ${connected ? '' : 'text-red'}`}>
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div className="meta-item">
            <span>Sim time</span>
            <span className="meta-value">{step.toLocaleString()}</span>
          </div>
          <div className="meta-item">
            <span>Engine</span>
            <span className="meta-value">{status?.device ?? '-'}</span>
          </div>
          <div className="meta-item">
            <span>Viewers</span>
            <span className="meta-value">{status?.websocket_clients ?? 0}</span>
          </div>
        </div>

        <div className="header-controls">
          <span className={`badge ${isRunning ? 'badge-live' : 'badge-stopped'}`}>
            <span className="badge-dot" />
            {isRunning ? 'Running' : 'Stopped'}
          </span>

          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>AI control</span>
            <label className="toggle">
              <input type="checkbox" checked={aiOn} onChange={handleToggleAI} disabled={!isRunning} />
              <span className="toggle-slider" />
            </label>
          </div>

          {!isRunning ? (
            <button className="btn btn-primary btn-sm" onClick={handleStart} disabled={loading}>
              {loading ? <span className="spinner" /> : '>'} Start
            </button>
          ) : (
            <button className="btn btn-danger btn-sm" onClick={stopSim} disabled={loading}>
              Stop
            </button>
          )}
          <button className="btn btn-ghost btn-sm" onClick={resetSim} disabled={loading}>
            Reset
          </button>
        </div>
      </header>

      <aside className="sidebar">
        <OverviewPanel status={status} connected={connected} summary={summary} alerts={alerts} />
        <KPIPanel summary={summary} vehicleCounts={vehicleCounts} />
        <AlertPanel alerts={alerts} />
      </aside>

      <LiveMap heatmap={heatmap} signalStates={signalStates} />

      <aside className="sidebar sidebar-right">
        <SignalMonitor signals={signalStates?.signals} alerts={alerts} actions={bundle?.actions} overrideSignal={overrideSignal} />
        <TrafficCharts history={history} />
      </aside>
    </div>
  )
}
