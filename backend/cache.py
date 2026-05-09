"""
backend/cache.py — Redis Cache Layer
=====================================
Centralised Redis interface for the Smart City Traffic AI backend.

Responsibilities:
  • Store live telemetry snapshots (TTL 60s)
  • Cache per-intersection state with fast O(1) reads
  • Maintain rolling history ring-buffer (last 500 snapshots)
  • Pub/Sub channel for multi-process fanout
  • Leaderboard of most-congested junctions (Redis Sorted Set)
  • Session state for simulation runs
  • Atomic KPI aggregation (counters, averages)

Key schema:
  traffic:state                 HASH   — current sim summary
  traffic:tls:{id}              HASH   — per-junction state
  traffic:history               LIST   — rolling 500 snapshots (JSON)
  traffic:heatmap               STRING — latest heatmap JSON blob
  traffic:alerts                LIST   — active alert objects
  traffic:congestion_rank       ZSET   — tls_id ranked by queue length
  traffic:kpi:vehicles          STRING — total vehicle count
  traffic:kpi:queue             STRING — total queue length
  traffic:kpi:wait              STRING — total waiting time
  traffic:session:{id}          HASH   — simulation session metadata
  traffic:pubsub                CHANNEL— real-time broadcast channel
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis
from redis.asyncio.client import PubSub

log = logging.getLogger("cache")

# ── Key constants ─────────────────────────────────────────────────────────────
K_STATE          = "traffic:state"
K_TLS_PREFIX     = "traffic:tls:"
K_HISTORY        = "traffic:history"
K_HEATMAP        = "traffic:heatmap"
K_ALERTS         = "traffic:alerts"
K_CONGESTION     = "traffic:congestion_rank"
K_KPI_VEHICLES   = "traffic:kpi:vehicles"
K_KPI_QUEUE      = "traffic:kpi:queue"
K_KPI_WAIT       = "traffic:kpi:wait"
K_SESSION_PREFIX = "traffic:session:"
K_PUBSUB_CHANNEL = "traffic:pubsub"
K_METRICS_TS     = "traffic:metrics_ts"     # time-series list for chart

# ── TTLs (seconds) ────────────────────────────────────────────────────────────
TTL_STATE    = 60
TTL_TLS      = 30
TTL_HEATMAP  = 5
TTL_ALERTS   = 120
TTL_KPIS     = 10
TTL_SESSION  = 7200     # 2 hours

# ── History / ring-buffer size ────────────────────────────────────────────────
HISTORY_MAX  = 500
METRICS_TS_MAX = 300   # 5 minutes at 1 pt/sec


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Manager
# ═══════════════════════════════════════════════════════════════════════════════
class TrafficCache:
    """
    Async Redis cache manager.
    All methods are safe to call even when Redis is unavailable
    (they silently return None / empty defaults).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._url    = redis_url
        self._redis: Optional[aioredis.Redis] = None
        self._ok     = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to Redis. Returns True if successful."""
        try:
            self._redis = await aioredis.from_url(
                self._url,
                decode_responses = True,
                socket_timeout   = 2.0,
                socket_connect_timeout = 2.0,
                retry_on_timeout = True,
                max_connections  = 20,
            )
            await self._redis.ping()
            self._ok = True
            log.info(f"Redis connected → {self._url}")
            return True
        except Exception as e:
            log.warning(f"Redis unavailable ({e}) — cache disabled")
            self._ok = False
            return False

    async def close(self):
        if self._redis:
            await self._redis.aclose()
            self._ok = False
            log.info("Redis disconnected")

    @property
    def available(self) -> bool:
        return self._ok and self._redis is not None

    def _r(self) -> Optional[aioredis.Redis]:
        return self._redis if self._ok else None

    # ── Simulation state ──────────────────────────────────────────────────────

    async def set_state(self, telemetry: Dict) -> None:
        """Write the current simulation summary as a flat Redis HASH."""
        r = self._r()
        if not r:
            return
        try:
            flat = {
                "step":           str(telemetry.get("step", 0)),
                "time_s":         str(telemetry.get("time_s", 0.0)),
                "total_vehicles": str(telemetry.get("total_vehicles", 0)),
                "total_queue":    str(telemetry.get("total_queue", 0)),
                "total_waiting":  str(telemetry.get("total_waiting", 0.0)),
                "ai_enabled":     str(telemetry.get("ai_enabled", True)),
                "device":         str(telemetry.get("device", "cpu")),
                "updated_at":     datetime.utcnow().isoformat(),
            }
            async with r.pipeline(transaction=False) as pipe:
                pipe.hset(K_STATE, mapping=flat)
                pipe.expire(K_STATE, TTL_STATE)
                await pipe.execute()
        except Exception as e:
            log.debug(f"set_state error: {e}")

    async def get_state(self) -> Optional[Dict]:
        """Read current simulation summary."""
        r = self._r()
        if not r:
            return None
        try:
            data = await r.hgetall(K_STATE)
            if not data:
                return None
            # Cast numeric fields
            for key in ("step", "total_vehicles", "total_queue"):
                if key in data:
                    data[key] = int(data[key])
            for key in ("time_s", "total_waiting"):
                if key in data:
                    data[key] = float(data[key])
            data["ai_enabled"] = data.get("ai_enabled", "True") == "True"
            return data
        except Exception as e:
            log.debug(f"get_state error: {e}")
            return None

    # ── Per-intersection cache ────────────────────────────────────────────────

    async def set_intersection(self, tls_id: str, data: Dict) -> None:
        """Cache a single junction's state. O(1) read by tls_id."""
        r = self._r()
        if not r:
            return
        try:
            key  = K_TLS_PREFIX + tls_id
            flat = {
                "phase_index":    str(data.get("phase_index", -1)),
                "phase_state":    str(data.get("phase_state", "")),
                "phase_duration": str(data.get("phase_duration", 0.0)),
                "total_queue":    str(data.get("total_queue", 0)),
                "total_vehicles": str(data.get("total_vehicles", 0)),
                "total_waiting":  str(data.get("total_waiting", 0.0)),
                "lane_count":     str(len(data.get("lanes", {}))),
                "updated_at":     datetime.utcnow().isoformat(),
            }
            async with r.pipeline(transaction=False) as pipe:
                pipe.hset(key, mapping=flat)
                pipe.expire(key, TTL_TLS)
                await pipe.execute()
        except Exception as e:
            log.debug(f"set_intersection({tls_id}) error: {e}")

    async def get_intersection(self, tls_id: str) -> Optional[Dict]:
        """Fast O(1) read for a single junction."""
        r = self._r()
        if not r:
            return None
        try:
            data = await r.hgetall(K_TLS_PREFIX + tls_id)
            if not data:
                return None
            for k in ("phase_index", "total_queue", "total_vehicles", "lane_count"):
                if k in data:
                    data[k] = int(data[k])
            for k in ("phase_duration", "total_waiting"):
                if k in data:
                    data[k] = float(data[k])
            return data
        except Exception as e:
            log.debug(f"get_intersection error: {e}")
            return None

    async def set_all_intersections(self, intersections: Dict[str, Dict]) -> None:
        """Bulk-write all intersection states in a single pipeline."""
        r = self._r()
        if not r:
            return
        try:
            async with r.pipeline(transaction=False) as pipe:
                for tls_id, data in intersections.items():
                    key  = K_TLS_PREFIX + tls_id
                    flat = {
                        "phase_index":    str(data.get("phase_index", -1)),
                        "phase_state":    str(data.get("phase_state", "")),
                        "phase_duration": str(data.get("phase_duration", 0.0)),
                        "total_queue":    str(data.get("total_queue", 0)),
                        "total_vehicles": str(data.get("total_vehicles", 0)),
                        "total_waiting":  str(data.get("total_waiting", 0.0)),
                        "updated_at":     datetime.utcnow().isoformat(),
                    }
                    pipe.hset(key, mapping=flat)
                    pipe.expire(key, TTL_TLS)
                await pipe.execute()
        except Exception as e:
            log.debug(f"set_all_intersections error: {e}")

    # ── History ring-buffer ───────────────────────────────────────────────────

    async def push_history(self, snapshot: Dict) -> None:
        """Push one telemetry snapshot; trim to HISTORY_MAX."""
        r = self._r()
        if not r:
            return
        try:
            payload = json.dumps({
                "step":      snapshot.get("step"),
                "time_s":    snapshot.get("time_s"),
                "vehicles":  snapshot.get("total_vehicles"),
                "queue":     snapshot.get("total_queue"),
                "waiting":   snapshot.get("total_waiting"),
                "ts":        datetime.utcnow().isoformat(),
            })
            async with r.pipeline(transaction=False) as pipe:
                pipe.lpush(K_HISTORY, payload)
                pipe.ltrim(K_HISTORY, 0, HISTORY_MAX - 1)
                await pipe.execute()
        except Exception as e:
            log.debug(f"push_history error: {e}")

    async def get_history(self, n: int = 100) -> List[Dict]:
        """Return last n history snapshots (newest first)."""
        r = self._r()
        if not r:
            return []
        try:
            n = min(n, HISTORY_MAX)
            raw = await r.lrange(K_HISTORY, 0, n - 1)
            return [json.loads(item) for item in raw]
        except Exception as e:
            log.debug(f"get_history error: {e}")
            return []

    async def clear_history(self) -> None:
        r = self._r()
        if r:
            try:
                await r.delete(K_HISTORY)
            except Exception:
                pass

    # ── Heatmap blob ──────────────────────────────────────────────────────────

    async def set_heatmap(self, heatmap_data: Dict) -> None:
        """Cache the latest heatmap payload (short TTL — always fresh)."""
        r = self._r()
        if not r:
            return
        try:
            await r.set(K_HEATMAP, json.dumps(heatmap_data), ex=TTL_HEATMAP)
        except Exception as e:
            log.debug(f"set_heatmap error: {e}")

    async def get_heatmap(self) -> Optional[Dict]:
        r = self._r()
        if not r:
            return None
        try:
            raw = await r.get(K_HEATMAP)
            return json.loads(raw) if raw else None
        except Exception as e:
            log.debug(f"get_heatmap error: {e}")
            return None

    # ── Congestion leaderboard (Sorted Set) ───────────────────────────────────

    async def update_congestion_rank(self, intersections: Dict[str, Dict]) -> None:
        """
        Update Redis Sorted Set: score = total_queue.
        Use ZADD to keep real-time leaderboard of most-congested junctions.
        """
        r = self._r()
        if not r:
            return
        try:
            mapping = {
                tls_id: float(data.get("total_queue", 0))
                for tls_id, data in intersections.items()
            }
            if mapping:
                await r.zadd(K_CONGESTION, mapping)
                await r.expire(K_CONGESTION, TTL_STATE)
        except Exception as e:
            log.debug(f"update_congestion_rank error: {e}")

    async def get_congestion_leaderboard(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return top N most congested junctions (tls_id, queue_score) descending."""
        r = self._r()
        if not r:
            return []
        try:
            results = await r.zrevrange(K_CONGESTION, 0, top_n - 1, withscores=True)
            return [(tls_id, score) for tls_id, score in results]
        except Exception as e:
            log.debug(f"get_congestion_leaderboard error: {e}")
            return []

    # ── KPI counters ──────────────────────────────────────────────────────────

    async def update_kpis(
        self,
        vehicles: int,
        queue:    int,
        wait:     float,
    ) -> None:
        """Atomic KPI update — fastest possible write path."""
        r = self._r()
        if not r:
            return
        try:
            async with r.pipeline(transaction=False) as pipe:
                pipe.set(K_KPI_VEHICLES, vehicles, ex=TTL_KPIS)
                pipe.set(K_KPI_QUEUE,    queue,    ex=TTL_KPIS)
                pipe.set(K_KPI_WAIT,     wait,     ex=TTL_KPIS)
                await pipe.execute()
        except Exception as e:
            log.debug(f"update_kpis error: {e}")

    async def get_kpis(self) -> Dict[str, Any]:
        """Read current KPIs — sub-millisecond from Redis."""
        r = self._r()
        if not r:
            return {}
        try:
            vehicles, queue, wait = await r.mget(
                K_KPI_VEHICLES, K_KPI_QUEUE, K_KPI_WAIT
            )
            return {
                "total_vehicles": int(vehicles) if vehicles else 0,
                "total_queue":    int(queue)    if queue    else 0,
                "total_waiting":  float(wait)   if wait     else 0.0,
            }
        except Exception as e:
            log.debug(f"get_kpis error: {e}")
            return {}

    # ── Metrics time-series ───────────────────────────────────────────────────

    async def push_metrics_ts(self, step: int, queue: int, vehicles: int, wait: float) -> None:
        """Push a compact metrics point to the time-series list."""
        r = self._r()
        if not r:
            return
        try:
            point = json.dumps({"s": step, "q": queue, "v": vehicles, "w": round(wait, 1)})
            async with r.pipeline(transaction=False) as pipe:
                pipe.rpush(K_METRICS_TS, point)
                pipe.ltrim(K_METRICS_TS, -METRICS_TS_MAX, -1)
                await pipe.execute()
        except Exception as e:
            log.debug(f"push_metrics_ts error: {e}")

    async def get_metrics_ts(self, n: int = 60) -> List[Dict]:
        """Return last n time-series points for chart rendering."""
        r = self._r()
        if not r:
            return []
        try:
            raw = await r.lrange(K_METRICS_TS, -n, -1)
            return [json.loads(p) for p in raw]
        except Exception as e:
            log.debug(f"get_metrics_ts error: {e}")
            return []

    # ── Alerts ────────────────────────────────────────────────────────────────

    async def set_alerts(self, alerts: List[Dict]) -> None:
        """Replace active alerts list atomically."""
        r = self._r()
        if not r:
            return
        try:
            async with r.pipeline(transaction=True) as pipe:
                pipe.delete(K_ALERTS)
                if alerts:
                    pipe.rpush(K_ALERTS, *[json.dumps(a) for a in alerts])
                    pipe.expire(K_ALERTS, TTL_ALERTS)
                await pipe.execute()
        except Exception as e:
            log.debug(f"set_alerts error: {e}")

    async def get_alerts(self) -> List[Dict]:
        """Return all current active alerts."""
        r = self._r()
        if not r:
            return []
        try:
            raw = await r.lrange(K_ALERTS, 0, -1)
            return [json.loads(a) for a in raw]
        except Exception as e:
            log.debug(f"get_alerts error: {e}")
            return []

    # ── Session management ────────────────────────────────────────────────────

    async def create_session(self, session_id: str, metadata: Dict) -> None:
        """Record a new simulation session."""
        r = self._r()
        if not r:
            return
        try:
            flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    for k, v in metadata.items()}
            flat["created_at"] = datetime.utcnow().isoformat()
            flat["session_id"] = session_id
            async with r.pipeline(transaction=False) as pipe:
                pipe.hset(K_SESSION_PREFIX + session_id, mapping=flat)
                pipe.expire(K_SESSION_PREFIX + session_id, TTL_SESSION)
                await pipe.execute()
            log.info(f"Session created: {session_id}")
        except Exception as e:
            log.debug(f"create_session error: {e}")

    async def update_session(self, session_id: str, updates: Dict) -> None:
        r = self._r()
        if not r:
            return
        try:
            flat = {k: str(v) for k, v in updates.items()}
            flat["updated_at"] = datetime.utcnow().isoformat()
            await r.hset(K_SESSION_PREFIX + session_id, mapping=flat)
        except Exception as e:
            log.debug(f"update_session error: {e}")

    async def get_session(self, session_id: str) -> Optional[Dict]:
        r = self._r()
        if not r:
            return None
        try:
            return await r.hgetall(K_SESSION_PREFIX + session_id) or None
        except Exception as e:
            log.debug(f"get_session error: {e}")
            return None

    # ── Pub/Sub ───────────────────────────────────────────────────────────────

    async def publish(self, message: Dict) -> int:
        """Publish a message to the traffic pubsub channel. Returns subscriber count."""
        r = self._r()
        if not r:
            return 0
        try:
            return await r.publish(K_PUBSUB_CHANNEL, json.dumps(message))
        except Exception as e:
            log.debug(f"publish error: {e}")
            return 0

    async def subscribe(self) -> Optional[PubSub]:
        """Return a PubSub object subscribed to the traffic channel."""
        r = self._r()
        if not r:
            return None
        try:
            pubsub = r.pubsub()
            await pubsub.subscribe(K_PUBSUB_CHANNEL)
            return pubsub
        except Exception as e:
            log.debug(f"subscribe error: {e}")
            return None

    # ── Full telemetry write (called every 500ms by streamer) ─────────────────

    async def write_telemetry(self, telemetry: Dict, heatmap: Dict, alerts: List[Dict]) -> None:
        """
        Single entry point for the streamer to write all cache data.
        Uses pipelines for minimal round-trips.
        """
        if not self.available:
            return

        intersections = telemetry.get("intersections", {})
        step          = telemetry.get("step", 0)
        vehicles      = telemetry.get("total_vehicles", 0)
        queue         = telemetry.get("total_queue", 0)
        wait          = telemetry.get("total_waiting", 0.0)

        # All writes fire concurrently
        await asyncio.gather(
            self.set_state(telemetry),
            self.set_all_intersections(intersections),
            self.push_history(telemetry),
            self.set_heatmap(heatmap),
            self.update_congestion_rank(intersections),
            self.update_kpis(vehicles, queue, wait),
            self.push_metrics_ts(step, queue, vehicles, wait),
            self.set_alerts(alerts),
            return_exceptions=True,
        )

    # ── Flush ─────────────────────────────────────────────────────────────────

    async def flush_simulation_data(self) -> None:
        """Clear all simulation-related keys (called on reset)."""
        r = self._r()
        if not r:
            return
        try:
            pattern_keys = []
            async for key in r.scan_iter("traffic:tls:*"):
                pattern_keys.append(key)

            keys_to_delete = [
                K_STATE, K_HISTORY, K_HEATMAP, K_ALERTS,
                K_CONGESTION, K_KPI_VEHICLES, K_KPI_QUEUE,
                K_KPI_WAIT, K_METRICS_TS,
            ] + pattern_keys

            if keys_to_delete:
                await r.delete(*keys_to_delete)
            log.info(f"Flushed {len(keys_to_delete)} cache keys")
        except Exception as e:
            log.warning(f"flush_simulation_data error: {e}")

    # ── Diagnostics ───────────────────────────────────────────────────────────

    async def info(self) -> Dict:
        """Return Redis server info + key counts for diagnostics endpoint."""
        r = self._r()
        if not r:
            return {"available": False}
        try:
            server_info = await r.info("server")
            memory_info = await r.info("memory")
            history_len = await r.llen(K_HISTORY)
            alerts_len  = await r.llen(K_ALERTS)
            tls_keys    = 0
            async for _ in r.scan_iter("traffic:tls:*"):
                tls_keys += 1

            return {
                "available":       True,
                "redis_version":   server_info.get("redis_version"),
                "used_memory_mb":  round(memory_info.get("used_memory", 0) / 1e6, 2),
                "history_entries": history_len,
                "active_alerts":   alerts_len,
                "tls_keys_cached": tls_keys,
                "url":             self._url,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}


import asyncio   # needed for write_telemetry gather (placed at bottom to avoid circular)


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton instance (imported by main.py and streamer.py)
# ═══════════════════════════════════════════════════════════════════════════════
import os
cache = TrafficCache(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))
