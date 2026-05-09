/**
 * TrafficCharts — live history charts only.
 */
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { useMemo } from 'react'

const CHART_STYLE = {
  fontSize: 10,
  fontFamily: 'JetBrains Mono, monospace',
}

const TOOLTIP_STYLE = {
  background: 'rgba(7,11,20,0.95)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 6,
  fontSize: 11,
  color: '#e8eaf0',
}

export default function TrafficCharts({ history = [] }) {
  // Build chart-ready arrays from rolling history
  const chartData = useMemo(() => {
    return history.slice(-40).map((h, i) => ({
      t:       h.step ?? i,
      queue:   h.queue   ?? h.q ?? 0,
      vehicles:h.vehicles ?? h.v ?? 0,
      wait:    +(h.waiting ?? h.w ?? 0).toFixed(1),
    }))
  }, [history])

  if (chartData.length === 0) {
    return (
      <div className="card">
      <div className="card-title"><span className="dot" />Trends Over Time</div>
        <div className="no-data">
          <strong>No chart data yet.</strong>
          <span>Start the simulation and this panel will show traffic changing over time.</span>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-title"><span className="dot" />Trends Over Time</div>

      <div className="chart-wrap-tall">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} style={CHART_STYLE}>
            <defs>
              <linearGradient id="gVeh" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#00d2ff" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00d2ff" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gQ" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#ff8c42" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ff8c42" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="t" tick={{ fill: '#4a5568' }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fill: '#4a5568' }} tickLine={false} axisLine={false} width={28} />
            <Tooltip contentStyle={TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <Area type="monotone" dataKey="vehicles" stroke="#00d2ff" fill="url(#gVeh)" strokeWidth={1.5} dot={false} name="Vehicles" />
            <Area type="monotone" dataKey="queue"    stroke="#ff8c42" fill="url(#gQ)"   strokeWidth={1.5} dot={false} name="Queue" />
            <Area type="monotone" dataKey="wait"     stroke="#ffd93d" fill="transparent" strokeWidth={1.5} dot={false} name="Wait" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
