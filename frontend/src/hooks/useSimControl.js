/**
 * useSimControl — simulation REST API hook
 */
import { useState, useEffect } from 'react'

const API = 'http://localhost:8000'

export function useSimControl() {
  const [status,  setStatus]  = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchStatus = async () => {
    try {
      const r = await fetch(`${API}/api/status`)
      setStatus(await r.json())
    } catch { setStatus(null) }
  }

  useEffect(() => {
    fetchStatus()
    const t = setInterval(fetchStatus, 3000)
    return () => clearInterval(t)
  }, [])

  const startSim = async (ai_enabled = true) => {
    setLoading(true)
    await fetch(`${API}/api/simulation/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ai_enabled, max_steps: 3600 }),
    }).catch(() => {})
    setLoading(false)
    fetchStatus()
  }

  const stopSim = async () => {
    setLoading(true)
    await fetch(`${API}/api/simulation/stop`, { method: 'POST' }).catch(() => {})
    setLoading(false)
    fetchStatus()
  }

  const resetSim = async () => {
    setLoading(true)
    await fetch(`${API}/api/simulation/reset`, { method: 'POST' }).catch(() => {})
    setLoading(false)
    fetchStatus()
  }

  const toggleAI = async () => {
    await fetch(`${API}/api/simulation/toggle_ai`, { method: 'POST' }).catch(() => {})
    fetchStatus()
  }

  const overrideSignal = async (tls_id, action) => {
    await fetch(`${API}/api/action/${tls_id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, tls_id }),
    }).catch(() => {})
  }

  return { status, loading, startSim, stopSim, resetSim, toggleAI, overrideSignal }
}
