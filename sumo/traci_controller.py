"""
TraCI Controller — SUMO ↔ Python Bridge
=========================================
Connects to a running SUMO simulation via TraCI and extracts
real-time traffic data from every intersection and lane.

Extracted metrics:
  - Queue length        (vehicles halted per lane)
  - Waiting time        (total accumulated wait per lane, seconds)
  - Vehicle count       (total vehicles on each lane)
  - Average speed       (mean speed of vehicles per lane, m/s)
  - Signal phase        (current TLS phase index + phase definition)
  - Occupancy           (% of lane occupied by vehicles)

Usage:
  Headless:   python traci_controller.py
  With GUI:   python traci_controller.py --gui
"""

import os
import sys
import time
import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# ── Locate SUMO tools ────────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    # Common install locations
    for candidate in ["/usr/share/sumo/tools", "/opt/sumo/tools"]:
        if os.path.isdir(candidate):
            sys.path.append(candidate)
            break

try:
    import traci
    import traci.constants as tc
    import sumolib
except ImportError as e:
    sys.exit(f"[ERROR] Could not import TraCI/sumolib: {e}\n"
             "Make sure SUMO is installed and SUMO_HOME is set.")

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "dhanbad.sumocfg")
SIM_STEP_LEN = 1.0          # seconds per simulation step
MAX_STEPS    = 3600          # 1 hour of simulated time
PORT         = 8813          # TraCI port
LOG_INTERVAL = 50            # print summary every N steps


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class LaneMetrics:
    lane_id:       str
    queue_length:  int     = 0     # number of halted vehicles
    waiting_time:  float   = 0.0   # total waiting time (s)
    vehicle_count: int     = 0     # vehicles currently on lane
    avg_speed:     float   = 0.0   # mean speed (m/s)
    occupancy:     float   = 0.0   # % lane occupied (0–100)


@dataclass
class IntersectionMetrics:
    junction_id:     str
    tls_id:          Optional[str]  = None
    phase_index:     int            = -1
    phase_duration:  float          = 0.0   # seconds in current phase
    phase_state:     str            = ""    # e.g. "GrGr" — green/red/yellow
    lanes:           Dict[str, LaneMetrics] = field(default_factory=dict)

    # Aggregated convenience properties
    @property
    def total_queue(self) -> int:
        return sum(l.queue_length for l in self.lanes.values())

    @property
    def total_waiting_time(self) -> float:
        return sum(l.waiting_time for l in self.lanes.values())

    @property
    def total_vehicles(self) -> int:
        return sum(l.vehicle_count for l in self.lanes.values())

    @property
    def network_avg_speed(self) -> float:
        lanes = [l for l in self.lanes.values() if l.vehicle_count > 0]
        if not lanes:
            return 0.0
        return sum(l.avg_speed for l in lanes) / len(lanes)


# ── TraCI Controller ──────────────────────────────────────────────────────────
class TraCIController:
    """
    Main controller class. Starts/connects to SUMO, subscribes to
    per-lane and per-TLS events, and collects metrics every step.
    """

    def __init__(self, use_gui: bool = False):
        self.use_gui       = use_gui
        self.step          = 0
        self.tls_ids:  List[str] = []
        self.lane_ids: List[str] = []
        self.junction_to_tls: Dict[str, str] = {}   # junction_id → tls_id
        self.history: List[Dict] = []                # rolling snapshot log

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Launch SUMO and open the TraCI connection."""
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary,
            "-c",        SUMOCFG_PATH,
            "--step-length", str(SIM_STEP_LEN),
            "--no-warnings", "true",
            "--collision.action", "warn",
        ]
        print(f"[TraCI] Starting SUMO: {' '.join(cmd)}")
        traci.start(cmd, port=PORT)
        self._discover_network()
        self._subscribe()
        print(f"[TraCI] Connected. "
              f"{len(self.tls_ids)} TLS junctions | "
              f"{len(self.lane_ids)} lanes")

    def close(self):
        """Gracefully close the TraCI connection."""
        traci.close()
        print(f"[TraCI] Closed after {self.step} steps.")

    # ── Network discovery ────────────────────────────────────────────────────

    def _discover_network(self):
        """Cache all TLS and lane IDs from the loaded network."""
        self.tls_ids  = list(traci.trafficlight.getIDList())
        self.lane_ids = list(traci.lane.getIDList())

        # Build junction → tls mapping
        for tls_id in self.tls_ids:
            # The controlled lanes tell us which junction this TLS belongs to
            ctrl_links = traci.trafficlight.getControlledLinks(tls_id)
            for link_group in ctrl_links:
                for link in link_group:
                    if link:                         # link = (in, out, via)
                        # The TLS ID is usually the same as junction ID in SUMO
                        self.junction_to_tls[tls_id] = tls_id
                        break

    def _subscribe(self):
        """
        Subscribe to per-lane variables for efficient bulk retrieval.
        Subscriptions are much faster than individual getXxx() calls.
        """
        lane_vars = [
            tc.LAST_STEP_VEHICLE_HALTING_NUMBER,   # queue length
            tc.LAST_STEP_VEHICLE_NUMBER,            # vehicle count
            tc.LAST_STEP_MEAN_SPEED,                # average speed
            tc.LAST_STEP_OCCUPANCY,                 # occupancy %
            tc.VAR_WAITING_TIME,                    # cumulative waiting time
        ]
        for lane_id in self.lane_ids:
            traci.lane.subscribe(lane_id, lane_vars)

        # Subscribe to TLS phase
        for tls_id in self.tls_ids:
            traci.trafficlight.subscribe(tls_id, [
                tc.TL_CURRENT_PHASE,
                tc.TL_PHASE_DURATION,
                tc.TL_RED_YELLOW_GREEN_STATE,
            ])

    # ── Metric extraction ────────────────────────────────────────────────────

    def collect_metrics(self) -> Dict[str, IntersectionMetrics]:
        """
        Pull subscription results for this step and return a dict of
        IntersectionMetrics keyed by TLS/junction ID.
        """
        # ── Lane data (bulk) ────────────────────────────────────────────────
        lane_results = traci.lane.getAllSubscriptionResults()

        lane_map: Dict[str, LaneMetrics] = {}
        for lane_id, vals in lane_results.items():
            m = LaneMetrics(lane_id=lane_id)
            m.queue_length  = vals.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            m.vehicle_count = vals.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            m.avg_speed     = vals.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
            m.occupancy     = vals.get(tc.LAST_STEP_OCCUPANCY, 0.0)
            m.waiting_time  = vals.get(tc.VAR_WAITING_TIME, 0.0)
            lane_map[lane_id] = m

        # ── TLS / intersection data ──────────────────────────────────────────
        tls_results = traci.trafficlight.getAllSubscriptionResults()

        intersections: Dict[str, IntersectionMetrics] = {}
        for tls_id in self.tls_ids:
            vals = tls_results.get(tls_id, {})
            im = IntersectionMetrics(junction_id=tls_id, tls_id=tls_id)
            im.phase_index    = vals.get(tc.TL_CURRENT_PHASE, -1)
            im.phase_duration = vals.get(tc.TL_PHASE_DURATION, 0.0)
            im.phase_state    = vals.get(tc.TL_RED_YELLOW_GREEN_STATE, "")

            # Attach controlled lanes to this intersection
            ctrl_lanes = set()
            for link_group in traci.trafficlight.getControlledLinks(tls_id):
                for link in link_group:
                    if link:
                        ctrl_lanes.add(link[0])   # incoming lane

            for lane_id in ctrl_lanes:
                if lane_id in lane_map:
                    im.lanes[lane_id] = lane_map[lane_id]

            intersections[tls_id] = im

        return intersections

    # ── Simulation loop ──────────────────────────────────────────────────────

    def run(self):
        """Main simulation loop — step through and collect metrics."""
        self.start()

        try:
            for step in range(MAX_STEPS):
                self.step = step
                traci.simulationStep()

                metrics = self.collect_metrics()
                snapshot = self._build_snapshot(step, metrics)
                self.history.append(snapshot)

                if step % LOG_INTERVAL == 0:
                    self._print_summary(step, metrics)

        except traci.exceptions.FatalTraCIError as exc:
            print(f"[TraCI] Simulation ended early: {exc}")
        finally:
            self.close()

        print(f"\n[TraCI] Collected {len(self.history)} snapshots.")
        return self.history

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_snapshot(
        self,
        step: int,
        metrics: Dict[str, IntersectionMetrics]
    ) -> Dict:
        """Serialise the current step's metrics to a plain dict."""
        return {
            "step": step,
            "time_s": step * SIM_STEP_LEN,
            "intersections": {
                jid: {
                    "tls_id":          im.tls_id,
                    "phase_index":     im.phase_index,
                    "phase_duration":  im.phase_duration,
                    "phase_state":     im.phase_state,
                    "total_queue":     im.total_queue,
                    "total_waiting":   round(im.total_waiting_time, 2),
                    "total_vehicles":  im.total_vehicles,
                    "avg_speed_ms":    round(im.network_avg_speed, 3),
                    "lanes": {
                        lid: {
                            "queue":       lm.queue_length,
                            "waiting":     round(lm.waiting_time, 2),
                            "vehicles":    lm.vehicle_count,
                            "speed_ms":    round(lm.avg_speed, 3),
                            "occupancy":   round(lm.occupancy, 2),
                        }
                        for lid, lm in im.lanes.items()
                    },
                }
                for jid, im in metrics.items()
            },
        }

    def _print_summary(
        self,
        step: int,
        metrics: Dict[str, IntersectionMetrics]
    ):
        """Print a one-line status summary to stdout."""
        total_vehicles  = traci.vehicle.getIDCount()
        total_queue     = sum(im.total_queue    for im in metrics.values())
        total_waiting   = sum(im.total_waiting_time for im in metrics.values())
        active_junctions = sum(1 for im in metrics.values()
                               if im.total_vehicles > 0)

        # Find the most congested intersection
        if metrics:
            worst = max(metrics.values(), key=lambda im: im.total_queue)
            worst_str = (f"{worst.junction_id[:20]:20s} "
                         f"queue={worst.total_queue:3d} "
                         f"wait={worst.total_waiting_time:6.1f}s")
        else:
            worst_str = "N/A"

        print(
            f"Step {step:4d} | "
            f"Vehicles: {total_vehicles:4d} | "
            f"Total queue: {total_queue:4d} | "
            f"Total wait: {total_waiting:8.1f}s | "
            f"Active TLS: {active_junctions:3d} | "
            f"Worst → {worst_str}"
        )

    def get_state_vector(
        self,
        tls_id: str,
        metrics: Dict[str, IntersectionMetrics]
    ) -> List[float]:
        """
        Build a flat state vector for a single intersection.
        Used as input to the RL agent in Phase 5.

        Features per lane (up to 8 lanes):
          [queue, waiting_time, vehicle_count, avg_speed, occupancy]
        + [phase_index_onehot × 4]

        Returns a fixed-length float list.
        """
        MAX_LANES   = 8
        MAX_PHASES  = 4
        LANE_FEATS  = 5

        im = metrics.get(tls_id)
        if im is None:
            return [0.0] * (MAX_LANES * LANE_FEATS + MAX_PHASES)

        lane_vec: List[float] = []
        for lane_id, lm in list(im.lanes.items())[:MAX_LANES]:
            lane_vec.extend([
                lm.queue_length  / 20.0,       # normalise (max ~20 vehicles)
                lm.waiting_time  / 200.0,       # normalise (max ~200 s)
                lm.vehicle_count / 20.0,
                lm.avg_speed     / 14.0,        # ~50 km/h = 13.9 m/s
                lm.occupancy     / 100.0,
            ])
        # Pad to fixed length
        lane_vec += [0.0] * (MAX_LANES * LANE_FEATS - len(lane_vec))

        # One-hot encode current phase
        phase_onehot = [0.0] * MAX_PHASES
        if 0 <= im.phase_index < MAX_PHASES:
            phase_onehot[im.phase_index] = 1.0

        return lane_vec + phase_onehot


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TraCI Controller — connect SUMO to Python"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch sumo-gui instead of headless sumo"
    )
    parser.add_argument(
        "--steps", type=int, default=MAX_STEPS,
        help=f"Number of simulation steps (default: {MAX_STEPS})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional JSON file to save collected snapshots"
    )
    args = parser.parse_args()

    MAX_STEPS = args.steps          # override global

    controller = TraCIController(use_gui=args.gui)
    history    = controller.run()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[TraCI] Saved {len(history)} snapshots → {args.output}")
