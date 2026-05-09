#!/usr/bin/env python3
"""
Measure WebSocket end-to-end latency for the live telemetry stream.

Run while the backend simulation is active:

    python scripts/test_websocket_latency.py --url ws://localhost:8000/ws/telemetry

The backend emits UTC ISO timestamps in each stream bundle. This script compares
that timestamp with local receipt time and fails if p95 exceeds the target.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone

import websockets


def parse_utc_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


async def measure_latency(url: str, samples: int, target_ms: float, timeout_s: float) -> int:
    latencies = []
    intervals = []
    last_received = None
    started = time.monotonic()

    async with websockets.connect(url, ping_interval=10, ping_timeout=10) as ws:
        while len(latencies) < samples:
            remaining = max(timeout_s - (time.monotonic() - started), 0.1)
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            received = datetime.now(timezone.utc)
            if last_received is not None:
                intervals.append((received - last_received).total_seconds() * 1000.0)
            last_received = received

            message = json.loads(raw)
            if message.get("type") != "stream_bundle" or "ts" not in message:
                continue

            emitted = parse_utc_iso(message["ts"])
            latency_ms = (received - emitted).total_seconds() * 1000.0
            if latency_ms >= 0:
                latencies.append(latency_ms)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[max(int(len(latencies) * 0.95) - 1, 0)]
    worst = max(latencies)
    avg_interval = statistics.mean(intervals) if intervals else 0.0

    print(f"samples={len(latencies)}")
    print(f"latency_ms p50={p50:.1f} p95={p95:.1f} max={worst:.1f}")
    print(f"stream_interval_ms avg={avg_interval:.1f}")
    print(f"target_p95_ms={target_ms:.1f}")

    if p95 > target_ms:
        print(f"FAIL: p95 latency {p95:.1f}ms exceeds {target_ms:.1f}ms", file=sys.stderr)
        return 1

    print("PASS: latency target satisfied")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure traffic dashboard WebSocket latency")
    parser.add_argument("--url", default="ws://localhost:8000/ws/telemetry")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--target-ms", type=float, default=200.0)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    args = parser.parse_args()

    return asyncio.run(measure_latency(args.url, args.samples, args.target_ms, args.timeout_s))


if __name__ == "__main__":
    raise SystemExit(main())
