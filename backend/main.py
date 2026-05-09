"""
backend/main.py — FastAPI Backend
==================================
Responsibilities:
  • Receive SUMO telemetry from TraCI controller
  • Run PPO prediction on RTX 4050 (CUDA)
  • Send signal actions back to SUMO
  • Stream live telemetry to frontend via WebSocket
  • Store analytics in Redis

Endpoints:
  GET  /                        health check
  GET  /api/status              simulation status
  GET  /api/intersections       all intersection states
  GET  /api/intersection/{id}   single intersection detail
  GET  /api/metrics             aggregated KPIs
  GET  /api/history?n=100       last N metric snapshots
  POST /api/simulation/start    start simulation + AI control
  POST /api/simulation/stop     stop simulation
  POST /api/simulation/reset    reset episode
  POST /api/action/{tls_id}     manual signal override
  WS   /ws/telemetry            real-time telemetry stream
  WS   /ws/simulation           simulation control stream
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager
from datetime import datetime

import torch
import numpy as np
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.streamer import SimulationStreamer
from backend.cache import cache as traffic_cache

# ── SUMO / TraCI path ─────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
AI_DIR      = PROJECT_DIR / "ai-engine"
SUMO_DIR    = PROJECT_DIR / "sumo"
sys.path.insert(0, str(AI_DIR))

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for _p in ["/usr/share/sumo/tools", "/opt/sumo/tools"]:
        if os.path.isdir(_p): sys.path.append(_p); break

try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("traffic-backend")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")

# ── Redis config ──────────────────────────────────────────────────────────────
REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_TTL     = 3600
HISTORY_KEY   = "traffic:history"
STATE_KEY     = "traffic:state"
METRICS_KEY   = "traffic:metrics"
MAX_HISTORY   = 500

# ── Simulation config ─────────────────────────────────────────────────────────
SUMOCFG_PATH  = SUMO_DIR / "dhanbad.sumocfg"
MODEL_PATH    = AI_DIR / "output" / "models" / "ppo_traffic_model.zip"
TRACI_PORT    = int(os.getenv("TRACI_PORT", "8850"))
TRACI_HOST    = os.getenv("TRACI_HOST")
TRACI_LABEL   = "backend_sim"
SIM_STEP_LEN  = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Application State
# ═══════════════════════════════════════════════════════════════════════════════
class SimulationState:
    def __init__(self):
        self.running:      bool  = False
        self.step:         int   = 0
        self.start_time:   Optional[float] = None
        self.tls_ids:      List[str] = []
        self.model        = None
        self.ai_enabled:  bool  = True
        self.current_obs:  Optional[np.ndarray] = None
        self.last_metrics: Dict  = {}
        self.device:       str   = str(DEVICE)   # ← exposed for streamer
        self.episode_stats: Dict = {
            "total_reward": 0.0,
            "total_queue":  [],
            "total_wait":   [],
        }

    def reset_stats(self):
        self.step       = 0
        self.start_time = time.time()
        self.episode_stats = {
            "total_reward": 0.0,
            "total_queue":  [],
            "total_wait":   [],
        }


sim_state  = SimulationState()
redis_client: Optional[aioredis.Redis] = None   # kept for compatibility
ws_manager: "ConnectionManager" = None


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket Connection Manager
# ═══════════════════════════════════════════════════════════════════════════════
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        log.info(f"WebSocket connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
        log.info(f"WebSocket disconnected. Total: {len(self.active)}")

    async def broadcast(self, data: dict):
        payload = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Model loader (GPU)
# ═══════════════════════════════════════════════════════════════════════════════
def load_ppo_model():
    """Load PPO model onto CUDA if available."""
    if not MODEL_PATH.exists():
        log.warning(f"No model found at {MODEL_PATH} — AI disabled")
        return None
    try:
        from stable_baselines3 import PPO
        model = PPO.load(
            str(MODEL_PATH),
            device = DEVICE,     # ← use RTX 4050
        )
        log.info(f"PPO model loaded on {DEVICE} ({sum(p.numel() for p in model.policy.parameters()):,} params)")
        return model
    except Exception as e:
        log.error(f"Failed to load PPO model: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SUMO / TraCI helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _start_sumo():
    if TRACI_HOST:
        log.info(f"Connecting to remote SUMO TraCI at {TRACI_HOST}:{TRACI_PORT}")
        traci.init(port=TRACI_PORT, host=TRACI_HOST, label=TRACI_LABEL)
    else:
        cmd = [
            "sumo", "-c", str(SUMOCFG_PATH),
            "--step-length",       str(SIM_STEP_LEN),
            "--no-warnings",       "true",
            "--collision.action",  "warn",
        ]
        traci.start(cmd, port=TRACI_PORT, label=TRACI_LABEL)
    traci.switch(TRACI_LABEL)
    sim_state.tls_ids = list(traci.trafficlight.getIDList())

    # Subscribe only controlled lanes; subscribing the full Dhanbad network
    # creates avoidable TraCI overhead and breaks the <200ms live target.
    controlled_lanes = set()
    for tid in sim_state.tls_ids:
        for group in traci.trafficlight.getControlledLinks(tid):
            for link in group:
                if link:
                    controlled_lanes.add(link[0])

    lane_vars = [
        tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
        tc.LAST_STEP_VEHICLE_NUMBER,
        tc.LAST_STEP_MEAN_SPEED,
        tc.LAST_STEP_OCCUPANCY,
        tc.VAR_WAITING_TIME,
    ]
    for lid in controlled_lanes:
        traci.lane.subscribe(lid, lane_vars)
    for tid in sim_state.tls_ids:
        traci.trafficlight.subscribe(tid, [
            tc.TL_CURRENT_PHASE,
            tc.TL_PHASE_DURATION,
            tc.TL_RED_YELLOW_GREEN_STATE,
        ])
    log.info(f"SUMO started — {len(sim_state.tls_ids)} TLS, {len(controlled_lanes)} controlled lanes subscribed")


def _stop_sumo():
    try:
        traci.switch(TRACI_LABEL)
        traci.close()
    except Exception:
        pass


def _collect_telemetry() -> Dict:
    """Pull one step of telemetry from SUMO subscriptions."""
    traci.switch(TRACI_LABEL)
    from emissions import emission_model

    lane_res = traci.lane.getAllSubscriptionResults()
    tls_res  = traci.trafficlight.getAllSubscriptionResults()

    intersections = {}
    for tid in sim_state.tls_ids:
        tv   = tls_res.get(tid, {})
        ctrl = traci.trafficlight.getControlledLinks(tid)
        ctrl_lanes = list({lnk[0] for g in ctrl for lnk in g if lnk})[:8]

        lanes_data = {}
        for lid in ctrl_lanes:
            v = lane_res.get(lid, {})
            class_counts = {}
            try:
                for vid in traci.lane.getLastStepVehicleIDs(lid):
                    try:
                        vehicle_class = traci.vehicle.getVehicleClass(vid)
                    except Exception:
                        vehicle_class = traci.vehicle.getTypeID(vid)
                    class_counts[vehicle_class] = class_counts.get(vehicle_class, 0) + 1
            except Exception:
                pass

            queue = v.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            vehicles = v.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            speed = round(v.get(tc.LAST_STEP_MEAN_SPEED, 0.0), 2)
            emission = emission_model.estimate_lane(
                speed_mps=speed,
                vehicle_count=vehicles,
                queue_count=queue,
                step_length_s=SIM_STEP_LEN,
                class_counts=class_counts,
            )
            lanes_data[lid] = {
                "queue":    queue,
                "vehicles": vehicles,
                "speed":    speed,
                "occupancy":round(v.get(tc.LAST_STEP_OCCUPANCY, 0.0), 2),
                "waiting":  round(v.get(tc.VAR_WAITING_TIME, 0.0), 2),
                "class_counts": class_counts,
                **emission,
            }

        total_co2_g = round(sum(l["co2_g"] for l in lanes_data.values()), 3)
        intersections[tid] = {
            "phase_index":   tv.get(tc.TL_CURRENT_PHASE, -1),
            "phase_duration":tv.get(tc.TL_PHASE_DURATION, 0.0),
            "phase_state":   tv.get(tc.TL_RED_YELLOW_GREEN_STATE, ""),
            "total_queue":   sum(l["queue"]    for l in lanes_data.values()),
            "total_vehicles":sum(l["vehicles"] for l in lanes_data.values()),
            "total_waiting": round(sum(l["waiting"]  for l in lanes_data.values()), 2),
            "total_co2_g":    total_co2_g,
            "co2_rate_g_s":   total_co2_g / SIM_STEP_LEN,
            "lanes":         lanes_data,
        }

    total_vehs = traci.vehicle.getIDCount()
    total_queue = sum(i["total_queue"]   for i in intersections.values())
    total_wait  = sum(i["total_waiting"] for i in intersections.values())
    total_co2_g = round(sum(i["total_co2_g"] for i in intersections.values()), 3)
    co2_rate_g_s = total_co2_g / SIM_STEP_LEN

    return {
        "step":          sim_state.step,
        "time_s":        sim_state.step * SIM_STEP_LEN,
        "timestamp":     datetime.utcnow().isoformat(),
        "total_vehicles":total_vehs,
        "total_queue":   total_queue,
        "total_waiting": round(total_wait, 2),
        "total_co2_g":   total_co2_g,
        "co2_rate_g_s":  round(co2_rate_g_s, 3),
        "co2_reduction_pct": emission_model.reduction_pct(co2_rate_g_s),
        "ai_enabled":    sim_state.ai_enabled,
        "device":        str(DEVICE),
        "intersections": intersections,
    }


def _build_obs_vector(telemetry: Dict, tls_id: str) -> np.ndarray:
    """Build normalised state vector for PPO inference (same shape as env)."""
    MAX_LANES, MAX_PHASES = 8, 8
    im = telemetry["intersections"].get(tls_id, {})
    obs = []
    for ld in list(im.get("lanes", {}).values())[:MAX_LANES]:
        obs.extend([
            min(ld["queue"]    / 20.0,  1.0),
            min(ld["waiting"]  / 200.0, 1.0),
            min(ld["vehicles"] / 20.0,  1.0),
            min(ld["speed"]    / 14.0,  1.0),
            ld["occupancy"] / 100.0,
        ])
    obs += [0.0] * (MAX_LANES * 5 - len(obs))
    phase_oh = [0.0] * MAX_PHASES
    pi = im.get("phase_index", 0)
    if 0 <= pi < MAX_PHASES:
        phase_oh[pi] = 1.0
    obs += phase_oh
    obs.append(min(im.get("phase_duration", 0) / 120.0, 1.0))
    obs.append(0.0)   # emergency flag
    return np.array(obs, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation background loop
# ═══════════════════════════════════════════════════════════════════════════════
def _apply_action_sync(telemetry: Dict) -> Dict:
    """
    Synchronous function (run in executor) that:
      1. Builds obs vector from telemetry
      2. Checks for Emergency Green Corridor override
      3. Runs per-junction PPO agents with TSP overrides
      4. Coordinates intersections for stable switching and green waves
      5. Applies the resulting actions to SUMO
    Returns dict of {tls_id: action} applied.
    """
    if not sim_state.model or not sim_state.tls_ids:
        return {}

    actions_taken = {}

    # -- Emergency Vehicle Green Corridor -----------------------------------
    try:
        traci.switch(TRACI_LABEL)
        from emergency import emergency_system
        emergency_activation = emergency_system.trigger_green_corridor(traci)
        if emergency_activation.active:
            emergency_payload = emergency_activation.to_dict()
            actions_taken.update(emergency_payload.get("applied_phases", {}))
            actions_taken["emergency_active"] = True
            actions_taken["emergency"] = emergency_payload
            actions_taken["bus_priority_active"] = False
            return actions_taken
    except Exception as e:
        log.debug(f"Emergency corridor error: {e}")

    # -- Multi-agent city coordination ---------------------------------------
    try:
        traci.switch(TRACI_LABEL)
        from multi_agent import multi_agent_coordinator
        with torch.no_grad():
            coordination = multi_agent_coordinator.control_step(
                traci=traci,
                telemetry=telemetry,
                tls_ids=sim_state.tls_ids,
                model=sim_state.model,
                obs_builder=_build_obs_vector,
            )
        actions_taken = coordination.to_dict()
        actions_taken["emergency_active"] = False
        return actions_taken
    except Exception as e:
        log.debug(f"Multi-agent coordination error: {e}")
        return {"multi_agent_active": False, "emergency_active": False, "error": str(e)}


async def simulation_loop():
    """Delegates to SimulationStreamer which streams at 500ms."""
    streamer = SimulationStreamer(
        sim_state   = sim_state,
        ws_manager  = ws_manager,
        redis_client= redis_client,
    )
    await streamer.run(
        collect_telemetry_fn = _collect_telemetry,
        apply_action_fn      = _apply_action_sync,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI app + lifespan
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, ws_manager
    ws_manager = ConnectionManager()

    # Connect cache (TrafficCache singleton)
    await traffic_cache.connect()

    # Keep a raw redis_client reference for legacy history endpoint
    if traffic_cache.available:
        redis_client = traffic_cache._redis
        log.info("Cache layer ready")
    else:
        log.warning("Cache layer unavailable — running without Redis")

    # Pre-load PPO model onto GPU
    sim_state.model = load_ppo_model()
    log.info("Backend ready")
    yield

    # Cleanup
    if sim_state.running:
        sim_state.running = False
        _stop_sumo()
    await traffic_cache.close()
    log.info("Backend shut down")


app = FastAPI(
    title       = "Smart City Traffic AI — Backend",
    description = "Real-time AI traffic signal control via SUMO + PPO (RTX 4050)",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════════════════════════════
class SimStartRequest(BaseModel):
    ai_enabled:    bool = True
    max_steps:     int  = 3600
    tls_id:        Optional[str] = None

class ActionRequest(BaseModel):
    action:  int    # 0=keep, 1=next, 2=skip, 3=emergency
    tls_id:  str


# ═══════════════════════════════════════════════════════════════════════════════
# REST Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    return {
        "service":   "Smart City Traffic AI",
        "status":    "online",
        "device":    str(DEVICE),
        "gpu":       torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "model_loaded": sim_state.model is not None,
        "sim_running":  sim_state.running,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/status", tags=["Simulation"])
async def get_status():
    uptime = round(time.time() - sim_state.start_time, 1) if sim_state.start_time else 0
    return {
        "running":    sim_state.running,
        "step":       sim_state.step,
        "uptime_s":   uptime,
        "ai_enabled": sim_state.ai_enabled,
        "tls_count":  len(sim_state.tls_ids),
        "device":     str(DEVICE),
        "websocket_clients": len(ws_manager.active) if ws_manager else 0,
    }


@app.get("/api/intersections", tags=["Traffic"])
async def get_intersections():
    if not sim_state.last_metrics:
        raise HTTPException(404, "No simulation data — start the simulation first")
    return {
        "step":          sim_state.last_metrics.get("step"),
        "intersections": sim_state.last_metrics.get("intersections", {}),
    }


@app.get("/api/intersection/{tls_id}", tags=["Traffic"])
async def get_intersection(tls_id: str):
    intersections = sim_state.last_metrics.get("intersections", {})
    if tls_id not in intersections:
        raise HTTPException(404, f"TLS '{tls_id}' not found")
    return {"tls_id": tls_id, **intersections[tls_id]}


@app.get("/api/metrics", tags=["Analytics"])
async def get_metrics():
    if redis_client:
        raw = await redis_client.get(METRICS_KEY)
        if raw:
            return json.loads(raw)

    # Fallback: live data
    if sim_state.last_metrics:
        return {
            "step":           sim_state.step,
            "total_vehicles": sim_state.last_metrics.get("total_vehicles", 0),
            "total_queue":    sim_state.last_metrics.get("total_queue", 0),
            "total_waiting":  sim_state.last_metrics.get("total_waiting", 0),
            "ai_enabled":     sim_state.ai_enabled,
            "device":         str(DEVICE),
        }
    raise HTTPException(404, "No metrics available yet")


@app.get("/api/history", tags=["Analytics"])
async def get_history(n: int = 100):
    history = await traffic_cache.get_history(n)
    if not history and not traffic_cache.available:
        raise HTTPException(503, "Redis not available")
    return {"count": len(history), "history": history}


@app.get("/api/metrics/timeseries", tags=["Analytics"])
async def get_metrics_timeseries(n: int = 60):
    """Return last N time-series points for chart rendering."""
    ts = await traffic_cache.get_metrics_ts(n)
    return {"count": len(ts), "series": ts}


@app.get("/api/congestion/leaderboard", tags=["Analytics"])
async def get_congestion_leaderboard(top: int = 5):
    """Return the most congested junctions ranked by queue length."""
    board = await traffic_cache.get_congestion_leaderboard(top)
    return {"leaderboard": [{"tls_id": t, "queue": s} for t, s in board]}


@app.get("/api/cache/info", tags=["Analytics"])
async def get_cache_info():
    """Redis diagnostics — memory, key counts, version."""
    return await traffic_cache.info()


@app.post("/api/simulation/start", tags=["Simulation"])
async def start_simulation(req: SimStartRequest, background_tasks: BackgroundTasks):
    if sim_state.running:
        raise HTTPException(400, "Simulation already running")
    if not TRACI_AVAILABLE:
        raise HTTPException(503, "TraCI not available — install SUMO")

    sim_state.ai_enabled = req.ai_enabled
    sim_state.reset_stats()

    # Start SUMO in a thread (blocking I/O)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _start_sumo)
    sim_state.running = True

    # Clear Redis history
    if redis_client:
        await redis_client.delete(HISTORY_KEY, STATE_KEY, METRICS_KEY)

    # Launch async simulation loop
    background_tasks.add_task(simulation_loop)

    log.info(f"Simulation started — AI={'ON' if req.ai_enabled else 'OFF'} | device={DEVICE}")
    return {
        "status":     "started",
        "ai_enabled": req.ai_enabled,
        "device":     str(DEVICE),
        "tls_count":  len(sim_state.tls_ids),
    }


@app.post("/api/simulation/stop", tags=["Simulation"])
async def stop_simulation():
    if not sim_state.running:
        raise HTTPException(400, "Simulation not running")
    sim_state.running = False
    await asyncio.sleep(0.2)
    return {"status": "stopped", "steps_completed": sim_state.step}


@app.post("/api/simulation/reset", tags=["Simulation"])
async def reset_simulation():
    was_running = sim_state.running
    if was_running:
        sim_state.running = False
        await asyncio.sleep(0.3)
    _stop_sumo()
    sim_state.step         = 0
    sim_state.last_metrics = {}
    if redis_client:
        await redis_client.delete(HISTORY_KEY, STATE_KEY, METRICS_KEY)
    return {"status": "reset"}


@app.post("/api/simulation/toggle_ai", tags=["Simulation"])
async def toggle_ai():
    sim_state.ai_enabled = not sim_state.ai_enabled
    return {"ai_enabled": sim_state.ai_enabled}


@app.post("/api/action/{tls_id}", tags=["Traffic"])
async def send_action(tls_id: str, req: ActionRequest):
    """Manual signal override — sends action directly to SUMO."""
    if not sim_state.running:
        raise HTTPException(400, "Simulation not running")
    if tls_id not in sim_state.tls_ids:
        raise HTTPException(404, f"TLS '{tls_id}' not found")
    if req.action not in (0, 1, 2, 3):
        raise HTTPException(422, "action must be 0-3")

    try:
        traci.switch(TRACI_LABEL)
        if req.action in (1, 2):
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            cur   = traci.trafficlight.getPhase(tls_id)
            n     = len(logic.phases)
            if n > 1:
                traci.trafficlight.setPhase(tls_id, (cur + req.action) % n)
        elif req.action == 3:
            state_len = len(traci.trafficlight.getRedYellowGreenState(tls_id))
            traci.trafficlight.setRedYellowGreenState(tls_id, "r" * state_len)
    except Exception as e:
        raise HTTPException(500, f"TraCI error: {e}")

    return {"tls_id": tls_id, "action_applied": req.action}


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    """Stream live telemetry to frontend at ~10 Hz."""
    await ws_manager.connect(ws)
    try:
        while True:
            # Keep connection alive; broadcast is handled by simulation_loop
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    """Control channel — receive commands, send confirmations."""
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                cmd = msg.get("command")
                if cmd == "ping":
                    await ws.send_json({"type": "pong", "step": sim_state.step})
                elif cmd == "status":
                    await ws.send_json({
                        "type":    "status",
                        "running": sim_state.running,
                        "step":    sim_state.step,
                        "ai":      sim_state.ai_enabled,
                        "device":  str(DEVICE),
                    })
                elif cmd == "toggle_ai":
                    sim_state.ai_enabled = not sim_state.ai_enabled
                    await ws.send_json({"type": "ai_toggled", "ai_enabled": sim_state.ai_enabled})
                else:
                    await ws.send_json({"type": "error", "msg": f"Unknown command: {cmd}"})
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "msg": "Invalid JSON"})
    except WebSocketDisconnect:
        pass
