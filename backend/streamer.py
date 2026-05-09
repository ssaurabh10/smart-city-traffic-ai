"""
backend/streamer.py — Real-time WebSocket Streaming Engine
===========================================================
Handles the full pipeline:
  SUMO → TraCI → PPO Prediction → FastAPI → WebSocket → Frontend

Stream interval : 500 ms
Payload types   : telemetry | heatmap | alerts | signal_states | summary
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

log = logging.getLogger("streamer")

# ── Congestion thresholds ─────────────────────────────────────────────────────
CONGESTION_QUEUE_THRESHOLD  = 5    # vehicles halted
CONGESTION_WAIT_THRESHOLD   = 60   # seconds
CONGESTION_SPEED_THRESHOLD  = 2.0  # m/s  (~7 km/h)
CRITICAL_QUEUE_THRESHOLD    = 15
HIGH_DEMAND_VEHICLE_THRESHOLD = 6
STREAM_INTERVAL             = 0.1  # 100 ms target for <200 ms end-to-end latency

# ── Alert severity ────────────────────────────────────────────────────────────
SEVERITY_INFO     = "info"
SEVERITY_WARNING  = "warning"
SEVERITY_CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class CongestionAlert:
    alert_id:    str
    tls_id:      str
    severity:    str
    message:     str
    queue:       int
    wait:        float
    timestamp:   str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class SignalState:
    tls_id:        str
    phase_index:   int
    phase_state:   str       # e.g. "GrGr"
    phase_duration:float
    is_green:      bool
    time_in_phase: float
    next_change_in:float     # estimated seconds until next phase


@dataclass
class HeatmapCell:
    lat:        float
    lon:        float
    intensity:  float        # 0.0–1.0
    queue:      int
    vehicles:   int
    tls_id:     str


# ═══════════════════════════════════════════════════════════════════════════════
# TLS coordinate lookup (from SUMO network — approximated from OSM bbox)
# Dhanbad bbox: lat 23.75–23.85, lon 86.40–86.50
# We distribute TLS across a realistic geographic area.
# In production, parse dhanbad.net.xml for exact junction coordinates.
# ═══════════════════════════════════════════════════════════════════════════════
_TLS_COORDS_CACHE: Dict[str, Tuple[float, float]] = {}

def _get_tls_coordinates(tls_id: str, index: int, total: int) -> Tuple[float, float]:
    """Return (lat, lon) for a TLS junction. Parses from SUMO or uses grid fallback."""
    if tls_id in _TLS_COORDS_CACHE:
        return _TLS_COORDS_CACHE[tls_id]

    # Try to get from TraCI junction position
    try:
        import traci
        x, y = traci.junction.getPosition(tls_id)
        # SUMO coordinates → geo
        lon, lat = traci.simulation.convertGeo(x, y, fromGeo=False)
        _TLS_COORDS_CACHE[tls_id] = (round(lat, 6), round(lon, 6))
        return _TLS_COORDS_CACHE[tls_id]
    except Exception:
        pass

    # Fallback: distribute in a grid over Dhanbad
    row   = index // 3
    col   = index  % 3
    lat   = 23.760 + row * 0.025
    lon   = 86.410 + col * 0.025
    _TLS_COORDS_CACHE[tls_id] = (round(lat, 6), round(lon, 6))
    return _TLS_COORDS_CACHE[tls_id]


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming payload builders
# ═══════════════════════════════════════════════════════════════════════════════
class StreamPayloadBuilder:
    """
    Converts raw SUMO telemetry into structured WebSocket payloads.
    Called every 500 ms by the simulation loop.
    """

    def __init__(self):
        self._alert_id     = 0
        self._active_alerts: Dict[str, CongestionAlert] = {}   # tls_id → alert
        self._prev_queues:   Dict[str, int]   = {}
        self._prev_speeds:   Dict[str, float] = {}
        self._history_queue: List[Dict]       = []   # rolling 60-point buffer
        self._step_times:    List[float]      = []   # for FPS calc

    # ── Vehicle count payload ─────────────────────────────────────────────────
    def build_vehicle_counts(self, telemetry: Dict) -> Dict:
        """Per-intersection vehicle / queue counts for dashboard counters."""
        counts = {}
        for tid, im in telemetry.get("intersections", {}).items():
            counts[tid] = {
                "vehicles":  im["total_vehicles"],
                "queue":     im["total_queue"],
                "waiting":   im["total_waiting"],
                "phase":     im["phase_index"],
            }
        return {
            "type":          "vehicle_counts",
            "step":          telemetry["step"],
            "time_s":        telemetry["time_s"],
            "total_vehicles":telemetry["total_vehicles"],
            "total_queue":   telemetry["total_queue"],
            "total_waiting": telemetry["total_waiting"],
            "intersections": counts,
        }

    # ── Heatmap payload ───────────────────────────────────────────────────────
    def build_heatmap(self, telemetry: Dict) -> Dict:
        """Geographic intensity heatmap for deck.gl / Mapbox."""
        cells = []
        tls_list = list(telemetry.get("intersections", {}).items())

        for i, (tid, im) in enumerate(tls_list):
            lat, lon = _get_tls_coordinates(tid, i, len(tls_list))
            max_q    = CRITICAL_QUEUE_THRESHOLD
            intensity = min(im["total_queue"] / max_q, 1.0)

            cells.append({
                "lat":       lat,
                "lon":       lon,
                "intensity": round(intensity, 3),
                "pollution_intensity": round(min(im.get("co2_rate_g_s", 0.0) / 75.0, 1.0), 3),
                "queue":     im["total_queue"],
                "vehicles":  im["total_vehicles"],
                "waiting":   im["total_waiting"],
                "co2_rate_g_s": round(im.get("co2_rate_g_s", 0.0), 2),
                "tls_id":    tid,
            })

        # Also add per-lane data for fine-grained heatmap
        lane_points = []
        for tid, im in tls_list:
            lat, lon = _get_tls_coordinates(tid, 0, 1)
            for lid, ld in list(im.get("lanes", {}).items())[:4]:
                intensity = min(ld["queue"] / 10.0, 1.0)
                lane_points.append({
                    "lat":       lat + (hash(lid) % 100) * 0.0001,
                    "lon":       lon + (hash(lid) % 50)  * 0.0001,
                    "intensity": round(intensity, 3),
                    "pollution_intensity": ld.get("emission_intensity", 0.0),
                    "queue":     ld["queue"],
                    "co2_rate_g_s": ld.get("co2_rate_g_s", 0.0),
                    "lane_id":   lid,
                })

        hotspots = sorted(
            cells,
            key=lambda cell: cell.get("co2_rate_g_s", 0.0),
            reverse=True,
        )[:5]

        return {
            "type":        "heatmap",
            "step":        telemetry["step"],
            "junctions":   cells,
            "lanes":       lane_points,
            "pollution_hotspots": hotspots,
        }

    # ── Signal states payload ─────────────────────────────────────────────────
    def build_signal_states(self, telemetry: Dict) -> Dict:
        """Current phase for every TLS — used to render signal icons."""
        signals = {}
        for tid, im in telemetry.get("intersections", {}).items():
            state_str = im.get("phase_state", "")
            has_green = any(c in ("G", "g") for c in state_str)
            phase_dur = im.get("phase_duration", 30.0)

            # Estimate time remaining in phase (phase_duration = elapsed time)
            # SUMO's TL_PHASE_DURATION is the *programmed* duration, not elapsed
            # We use our own timer approximation
            signals[tid] = {
                "phase_index":    im["phase_index"],
                "phase_state":    state_str,
                "phase_duration": phase_dur,
                "is_green":       has_green,
                "green_count":    state_str.count("G") + state_str.count("g"),
                "red_count":      state_str.count("r") + state_str.count("R"),
                "yellow_count":   state_str.count("y") + state_str.count("Y"),
            }

        return {
            "type":    "signal_states",
            "step":    telemetry["step"],
            "signals": signals,
        }

    # ── Congestion alerts payload ─────────────────────────────────────────────
    def build_alerts(self, telemetry: Dict) -> Dict:
        """Detect congestion events and emit/clear alerts."""
        new_alerts: List[Dict]     = []
        cleared_alerts: List[str]  = []
        current_tls_with_issue: Set[str] = set()

        for tid, im in telemetry.get("intersections", {}).items():
            queue   = im["total_queue"]
            waiting = im["total_waiting"]
            vehicles = im.get("total_vehicles", 0)

            # Determine severity
            if queue >= CRITICAL_QUEUE_THRESHOLD:
                severity = SEVERITY_CRITICAL
                msg      = f"CRITICAL: {queue} vehicles halted — {waiting:.0f}s avg wait"
            elif queue >= CONGESTION_QUEUE_THRESHOLD or waiting >= CONGESTION_WAIT_THRESHOLD:
                severity = SEVERITY_WARNING
                msg      = f"Congestion: {queue} vehicles queued — {waiting:.0f}s wait"
            elif vehicles >= HIGH_DEMAND_VEHICLE_THRESHOLD:
                severity = SEVERITY_WARNING
                msg      = f"High demand: {vehicles} vehicles on signal approach — AI monitoring"
            else:
                severity = None

            if severity:
                current_tls_with_issue.add(tid)
                prev = self._active_alerts.get(tid)
                # Only emit if new or severity escalated
                if not prev or prev.severity != severity:
                    self._alert_id += 1
                    alert = CongestionAlert(
                        alert_id = f"ALT-{self._alert_id:04d}",
                        tls_id   = tid,
                        severity = severity,
                        message  = msg,
                        queue    = queue,
                        wait     = round(waiting, 1),
                    )
                    self._active_alerts[tid] = alert
                    new_alerts.append(alert.to_dict())
            else:
                # Clear existing alert if congestion resolved
                if tid in self._active_alerts:
                    cleared_alerts.append(self._active_alerts.pop(tid).alert_id)

        return {
            "type":           "alerts",
            "step":           telemetry["step"],
            "new_alerts":     new_alerts,
            "cleared":        cleared_alerts,
            "active_count":   len(self._active_alerts),
            "active_alerts":  [a.to_dict() for a in self._active_alerts.values()],
        }

    # ── Rolling summary ───────────────────────────────────────────────────────
    def build_summary(self, telemetry: Dict, ai_enabled: bool, device: str) -> Dict:
        """Aggregated KPIs + trend for dashboard summary cards."""
        now = time.time()
        self._step_times.append(now)
        # Keep last 20 step times for FPS
        if len(self._step_times) > 20:
            self._step_times.pop(0)

        fps = 0.0
        if len(self._step_times) >= 2:
            fps = (len(self._step_times) - 1) / (self._step_times[-1] - self._step_times[0])

        # Rolling history
        self._history_queue.append({
            "step":    telemetry["step"],
            "queue":   telemetry["total_queue"],
            "vehicles":telemetry["total_vehicles"],
            "waiting": telemetry["total_waiting"],
            "co2_rate_g_s": telemetry.get("co2_rate_g_s", 0.0),
        })
        if len(self._history_queue) > 60:
            self._history_queue.pop(0)

        queue_trend = 0.0
        if len(self._history_queue) >= 10:
            recent = [h["queue"] for h in self._history_queue[-5:]]
            older  = [h["queue"] for h in self._history_queue[-10:-5]]
            queue_trend = np.mean(recent) - np.mean(older)

        return {
            "type":          "summary",
            "step":          telemetry["step"],
            "time_s":        telemetry["time_s"],
            "ai_enabled":    ai_enabled,
            "device":        device,
            "fps":           round(fps, 1),
            "total_vehicles":telemetry["total_vehicles"],
            "total_queue":   telemetry["total_queue"],
            "total_waiting": round(telemetry["total_waiting"], 1),
            "co2_rate_g_s":  round(telemetry.get("co2_rate_g_s", 0.0), 2),
            "total_co2_g":   round(telemetry.get("total_co2_g", 0.0), 2),
            "co2_reduction_pct": telemetry.get("co2_reduction_pct", 0.0),
            "queue_trend":   round(queue_trend, 2),   # + = worsening, - = improving
            "history":       self._history_queue[-30:],
            "active_alerts": len(self._active_alerts),
            "timestamp":     datetime.utcnow().isoformat(),
        }

    # ── Full bundle (single WS message every 500ms) ───────────────────────────
    def build_stream_bundle(
        self,
        telemetry:  Dict,
        ai_enabled: bool,
        device:     str,
    ) -> Dict:
        """
        Combine all payloads into one message sent every 500ms.
        Frontend unpacks by `type` fields nested inside.
        """
        return {
            "type":          "stream_bundle",
            "ts":            datetime.utcnow().isoformat(),
            "vehicle_counts":self.build_vehicle_counts(telemetry),
            "heatmap":       self.build_heatmap(telemetry),
            "signal_states": self.build_signal_states(telemetry),
            "alerts":        self.build_alerts(telemetry),
            "summary":       self.build_summary(telemetry, ai_enabled, device),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Async Streaming Loop (replaces simulation_loop in main.py)
# ═══════════════════════════════════════════════════════════════════════════════
class SimulationStreamer:
    """
    Drives the SUMO simulation at native speed, but only broadcasts
    to WebSocket clients every STREAM_INTERVAL (500ms).
    """

    def __init__(self, sim_state, ws_manager, redis_client=None):
        self.sim_state    = sim_state
        self.ws_manager   = ws_manager
        self.redis        = redis_client
        self.builder      = StreamPayloadBuilder()
        self._last_stream = 0.0
        self._step_buffer: List[Dict] = []   # accumulate steps between streams

    async def run(self, collect_telemetry_fn, apply_action_fn):
        """
        Main loop:
          1. Step SUMO (every call)
          2. Collect telemetry
          3. Run PPO inference
          4. If 500ms elapsed → build & broadcast bundle
        """
        log.info(f"[Streamer] Started — interval={STREAM_INTERVAL*1000:.0f}ms")

        try:
            while self.sim_state.running:
                t_start = time.time()

                # ── Step SUMO ─────────────────────────────────────────────────
                try:
                    import traci
                    traci.switch("backend_sim")
                    traci.simulationStep()
                    self.sim_state.step += 1
                except Exception as e:
                    log.error(f"SUMO step error: {e}")
                    break

                # ── Collect telemetry ─────────────────────────────────────────
                try:
                    telemetry = collect_telemetry_fn()
                    self.sim_state.last_metrics = telemetry
                except Exception as e:
                    log.warning(f"Telemetry collection error: {e}")
                    telemetry = {}

                # ── PPO inference + action ────────────────────────────────────
                action_result = {}
                if self.sim_state.ai_enabled and self.sim_state.model and telemetry:
                    try:
                        action_result = await asyncio.get_event_loop().run_in_executor(
                            None, apply_action_fn, telemetry
                        )
                    except Exception as e:
                        log.debug(f"Action apply error: {e}")

                # ── Buffer this step ──────────────────────────────────────────
                if telemetry:
                    self._step_buffer.append({
                        "step":    self.sim_state.step,
                        "queue":   telemetry.get("total_queue", 0),
                        "vehicles":telemetry.get("total_vehicles", 0),
                    })

                # ── Broadcast every 500ms ─────────────────────────────────────
                now = time.time()
                if (now - self._last_stream) >= STREAM_INTERVAL and telemetry:
                    self._last_stream = now

                    bundle = self.builder.build_stream_bundle(
                        telemetry  = telemetry,
                        ai_enabled = self.sim_state.ai_enabled,
                        device     = str(self.sim_state.device
                                         if hasattr(self.sim_state, "device")
                                         else "cuda"),
                    )
                    bundle["buffered_steps"] = len(self._step_buffer)
                    bundle["actions"]        = action_result
                    self._step_buffer.clear()

                    # Broadcast to all WebSocket clients
                    await self.ws_manager.broadcast(bundle)

                    # Persist to Redis
                    if self.redis:
                        await self._persist_to_redis(bundle, telemetry)

                # ── Yield control ──────────────────────────────────────────────
                elapsed = time.time() - t_start
                await asyncio.sleep(max(0.0, 0.01 - elapsed))   # ~100 SUMO steps/s cap

        except asyncio.CancelledError:
            log.info("[Streamer] Cancelled")
        except Exception as e:
            log.error(f"[Streamer] Fatal error: {e}", exc_info=True)
        finally:
            self.sim_state.running = False
            await self.ws_manager.broadcast({
                "type":           "simulation_stopped",
                "steps_completed":self.sim_state.step,
                "timestamp":      datetime.utcnow().isoformat(),
            })
            log.info(f"[Streamer] Stopped after {self.sim_state.step} steps")

    async def _persist_to_redis(self, bundle: Dict, telemetry: Dict):
        """Delegate all Redis writes to the TrafficCache singleton."""
        try:
            from backend.cache import cache as traffic_cache
            heatmap = bundle.get("heatmap", {})
            alerts  = bundle.get("alerts", {}).get("active_alerts", [])
            await traffic_cache.write_telemetry(telemetry, heatmap, alerts)
        except Exception as e:
            log.debug(f"Cache write error: {e}")
