/**
 * useTrafficStream — WebSocket hook
 * Connects to ws://localhost:8000/ws/telemetry
 * Parses stream_bundle messages every 500ms
 */
import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = 'ws://localhost:8000/ws/telemetry'
const RECONNECT_DELAY = 2000

export function useTrafficStream() {
  const [bundle,    setBundle]    = useState(null)
  const [connected, setConnected] = useState(false)
  const [error,     setError]     = useState(null)
  const wsRef = useRef(null)
  const timerRef = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      setError(null)
    }

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        if (msg.type === 'stream_bundle') setBundle(msg)
      } catch { /* ignore malformed */ }
    }

    ws.onerror = () => setError('WebSocket error')

    ws.onclose = () => {
      setConnected(false)
      timerRef.current = setTimeout(connect, RECONNECT_DELAY)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { bundle, connected, error }
}
