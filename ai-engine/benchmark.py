"""
Phase 20 benchmark runner.

Compares:
  * PPO RL multi-intersection control
  * Webster fixed-cycle timing
  * Static/default SUMO timing

Metrics:
  * waiting time
  * queue length
  * CO2 emissions
  * throughput
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for _p in ["/usr/share/sumo/tools", "/opt/sumo/tools"]:
        if os.path.isdir(_p):
            sys.path.append(_p)
            break

import traci
import traci.constants as tc

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SUMOCFG = PROJECT_DIR / "sumo" / "dhanbad.sumocfg"
MODEL_PATH = BASE_DIR / "output" / "models" / "ppo_traffic_model.zip"
RESULT_DIR = BASE_DIR / "output" / "results"
PLOT_DIR = BASE_DIR / "output" / "plots"
for directory in (RESULT_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SATURATION_FLOW = 1800
LOST_TIME_PER_PHASE = 4
MIN_CYCLE = 30
MAX_CYCLE = 120
MAX_LANES = 8
MAX_PHASES = 8


@dataclass
class BenchmarkMetrics:
    label: str
    queue: List[float] = field(default_factory=list)
    waiting: List[float] = field(default_factory=list)
    co2_mg_s: List[float] = field(default_factory=list)
    arrived: int = 0
    vehicles_seen: set = field(default_factory=set)

    @property
    def mean_queue(self) -> float:
        return float(np.mean(self.queue)) if self.queue else 0.0

    @property
    def mean_waiting_s(self) -> float:
        return float(np.mean(self.waiting)) if self.waiting else 0.0

    @property
    def mean_co2_mg_s(self) -> float:
        return float(np.mean(self.co2_mg_s)) if self.co2_mg_s else 0.0

    @property
    def throughput(self) -> int:
        return len(self.vehicles_seen)

    @property
    def vehicles_observed(self) -> int:
        return len(self.vehicles_seen)

    def to_dict(self) -> Dict:
        return {
            "mean_waiting_s": round(self.mean_waiting_s, 3),
            "mean_queue_length": round(self.mean_queue, 3),
            "mean_co2_mg_s": round(self.mean_co2_mg_s, 3),
            "mean_co2_g_s": round(self.mean_co2_mg_s / 1000.0, 3),
            "throughput_vehicles": self.throughput,
            "arrived_vehicles": int(self.arrived),
            "vehicles_observed": self.vehicles_observed,
            "samples": len(self.queue),
        }


def start_sumo(label: str, port: int, seed: int) -> None:
    cmd = [
        "sumo",
        "-c", str(SUMOCFG),
        "--step-length", "1",
        "--seed", str(seed),
        "--no-warnings", "true",
        "--collision.action", "warn",
    ]
    traci.start(cmd, port=port, label=label)
    traci.switch(label)


def stop_sumo(label: str) -> None:
    try:
        traci.switch(label)
        traci.close()
    except Exception:
        pass


def controlled_lanes_by_tls() -> Dict[str, List[str]]:
    mapping = {}
    for tls_id in traci.trafficlight.getIDList():
        lanes = []
        seen = set()
        for group in traci.trafficlight.getControlledLinks(tls_id):
            for link in group:
                if link and link[0] not in seen:
                    seen.add(link[0])
                    lanes.append(link[0])
        mapping[tls_id] = lanes[:MAX_LANES]
    return mapping


def subscribe_lanes(lane_ids: List[str]) -> None:
    lane_vars = [
        tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
        tc.LAST_STEP_VEHICLE_NUMBER,
        tc.LAST_STEP_MEAN_SPEED,
        tc.LAST_STEP_OCCUPANCY,
        tc.VAR_WAITING_TIME,
    ]
    for lane_id in lane_ids:
        traci.lane.subscribe(lane_id, lane_vars)


def build_telemetry(lanes_by_tls: Dict[str, List[str]], step: int) -> Dict:
    lane_res = traci.lane.getAllSubscriptionResults()
    intersections = {}

    for tls_id, lanes in lanes_by_tls.items():
        lanes_data = {}
        for lane_id in lanes:
            values = lane_res.get(lane_id, {})
            lanes_data[lane_id] = {
                "queue": values.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0),
                "vehicles": values.get(tc.LAST_STEP_VEHICLE_NUMBER, 0),
                "speed": values.get(tc.LAST_STEP_MEAN_SPEED, 0.0),
                "occupancy": values.get(tc.LAST_STEP_OCCUPANCY, 0.0),
                "waiting": values.get(tc.VAR_WAITING_TIME, 0.0),
            }

        phase_index = -1
        phase_state = ""
        phase_duration = 0.0
        try:
            phase_index = traci.trafficlight.getPhase(tls_id)
            phase_state = traci.trafficlight.getRedYellowGreenState(tls_id)
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            if 0 <= phase_index < len(logic.phases):
                phase_duration = logic.phases[phase_index].duration
        except Exception:
            pass

        intersections[tls_id] = {
            "phase_index": phase_index,
            "phase_duration": phase_duration,
            "phase_state": phase_state,
            "total_queue": sum(v["queue"] for v in lanes_data.values()),
            "total_vehicles": sum(v["vehicles"] for v in lanes_data.values()),
            "total_waiting": sum(v["waiting"] for v in lanes_data.values()),
            "lanes": lanes_data,
        }

    return {
        "step": step,
        "time_s": step,
        "total_vehicles": traci.vehicle.getIDCount(),
        "total_queue": sum(v["total_queue"] for v in intersections.values()),
        "total_waiting": sum(v["total_waiting"] for v in intersections.values()),
        "intersections": intersections,
    }


def build_obs_vector(telemetry: Dict, tls_id: str) -> np.ndarray:
    im = telemetry["intersections"].get(tls_id, {})
    obs = []
    for lane_data in list(im.get("lanes", {}).values())[:MAX_LANES]:
        obs.extend([
            min(lane_data["queue"] / 20.0, 1.0),
            min(lane_data["waiting"] / 200.0, 1.0),
            min(lane_data["vehicles"] / 20.0, 1.0),
            min(lane_data["speed"] / 14.0, 1.0),
            lane_data["occupancy"] / 100.0,
        ])
    obs += [0.0] * (MAX_LANES * 5 - len(obs))
    phase_oh = [0.0] * MAX_PHASES
    phase_index = im.get("phase_index", 0)
    if 0 <= phase_index < MAX_PHASES:
        phase_oh[phase_index] = 1.0
    obs += phase_oh
    obs.append(min(im.get("phase_duration", 0) / 120.0, 1.0))
    obs.append(0.0)
    return np.array(obs, dtype=np.float32)


def sample_metrics(label: str, metrics: BenchmarkMetrics, telemetry: Dict) -> None:
    vehicle_ids = traci.vehicle.getIDList()
    metrics.queue.append(sum(1 for vid in vehicle_ids if traci.vehicle.getSpeed(vid) < 0.1))
    metrics.waiting.append(sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids))
    metrics.co2_mg_s.append(sum(traci.vehicle.getCO2Emission(vid) for vid in vehicle_ids))
    metrics.arrived += traci.simulation.getArrivedNumber()
    metrics.vehicles_seen.update(vehicle_ids)


def apply_static_timing() -> None:
    """Leave SUMO default static programs untouched."""
    return


def apply_webster_timing(lanes_by_tls: Dict[str, List[str]], warmup_steps: int) -> None:
    flow_counts: Dict[str, List[int]] = {}
    for tls_id in traci.trafficlight.getIDList():
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        flow_counts[tls_id] = [0] * len(logic.phases)

    for _ in range(warmup_steps):
        traci.simulationStep()
        for tls_id, lanes in lanes_by_tls.items():
            phase = traci.trafficlight.getPhase(tls_id)
            try:
                flow_counts[tls_id][phase] += sum(
                    traci.lane.getLastStepVehicleNumber(lane_id)
                    for lane_id in lanes
                )
            except Exception:
                pass

    for tls_id, counts in flow_counts.items():
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        n_phases = max(len(logic.phases), 1)
        flows = [max(1.0, count * 3600.0 / max(warmup_steps, 1)) for count in counts]
        lost_time = n_phases * LOST_TIME_PER_PHASE
        ratios = [flow / SATURATION_FLOW for flow in flows]
        total_ratio = min(sum(ratios), 0.9)
        cycle = float(np.clip((1.5 * lost_time + 5) / (1 - total_ratio), MIN_CYCLE, MAX_CYCLE))
        effective_green = max(cycle - lost_time, n_phases * 5.0)
        greens = [
            (ratio / total_ratio) * effective_green if total_ratio > 0 else effective_green / n_phases
            for ratio in ratios
        ]

        phases = []
        for idx, phase in enumerate(logic.phases):
            duration = max(5.0, greens[idx] if idx < len(greens) else 10.0)
            phases.append(traci.trafficlight.Phase(duration, phase.state))

        try:
            new_logic = traci.trafficlight.Logic(
                programID="webster_benchmark",
                type=0,
                currentPhaseIndex=0,
                phases=phases,
            )
            traci.trafficlight.setProgramLogic(tls_id, new_logic)
            traci.trafficlight.setProgram(tls_id, "webster_benchmark")
        except Exception:
            pass


def run_strategy(
    strategy: str,
    label: str,
    steps: int,
    warmup_steps: int,
    port: int,
    seed: int,
):
    start_sumo(label=f"benchmark_{strategy}", port=port, seed=seed)
    try:
        lanes_by_tls = controlled_lanes_by_tls()
        controlled_lane_ids = sorted({lane for lanes in lanes_by_tls.values() for lane in lanes})
        subscribe_lanes(controlled_lane_ids)
        tls_ids = list(lanes_by_tls.keys())

        model = None
        coordinator = None
        if strategy == "rl":
            from stable_baselines3 import PPO
            from multi_agent import MultiIntersectionCoordinator

            model = PPO.load(str(MODEL_PATH), device="cpu")
            coordinator = MultiIntersectionCoordinator(
                min_switch_interval_s=8.0,
                max_switches_per_step=4,
                green_wave_queue_threshold=6,
            )
        elif strategy == "webster":
            apply_webster_timing(lanes_by_tls, warmup_steps)
        else:
            apply_static_timing()

        if strategy in ("rl", "static") and warmup_steps > 0:
            for warm_step in range(warmup_steps):
                traci.simulationStep()
                if strategy == "rl" and model is not None:
                    warm_telemetry = build_telemetry(lanes_by_tls, -warmup_steps + warm_step)
                    coordinator.control_step(
                        traci=traci,
                        telemetry=warm_telemetry,
                        tls_ids=tls_ids,
                        model=model,
                        obs_builder=build_obs_vector,
                    )

        metrics = BenchmarkMetrics(label=label)
        for step in range(steps):
            traci.simulationStep()
            telemetry = build_telemetry(lanes_by_tls, step)
            if strategy == "rl" and model is not None:
                coordinator.control_step(
                    traci=traci,
                    telemetry=telemetry,
                    tls_ids=tls_ids,
                    model=model,
                    obs_builder=build_obs_vector,
                )
            sample_metrics(label, metrics, telemetry)

            if (step + 1) % max(steps // 5, 1) == 0:
                print(
                    f"  [{label}] {step + 1:4d}/{steps} "
                    f"queue={metrics.mean_queue:.2f} "
                    f"wait={metrics.mean_waiting_s:.2f}s "
                    f"co2={metrics.mean_co2_mg_s:.0f}mg/s "
                    f"throughput={metrics.throughput}"
                )

        return metrics
    finally:
        stop_sumo(f"benchmark_{strategy}")


def improvement_table(results: Dict[str, BenchmarkMetrics]) -> Dict:
    rl = results.get("PPO RL Multi-Agent")
    if not rl:
        return {}

    improvements = {}
    for baseline_name in ("Webster Fixed-Cycle", "Static Timing"):
        baseline = results.get(baseline_name)
        if not baseline:
            continue
        improvements[baseline_name] = {
            "waiting_time_pct": pct_reduction(baseline.mean_waiting_s, rl.mean_waiting_s),
            "queue_length_pct": pct_reduction(baseline.mean_queue, rl.mean_queue),
            "emissions_pct": pct_reduction(baseline.mean_co2_mg_s, rl.mean_co2_mg_s),
            "throughput_pct": pct_increase(baseline.throughput, rl.throughput),
        }
    return improvements


def pct_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return round((baseline - candidate) / baseline * 100.0, 2)


def pct_increase(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return round((candidate - baseline) / baseline * 100.0, 2)


def save_plots(results: Dict[str, BenchmarkMetrics]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    values = {
        "Waiting Time (s)": [results[label].mean_waiting_s for label in labels],
        "Queue Length": [results[label].mean_queue for label in labels],
        "CO2 (g/s)": [results[label].mean_co2_mg_s / 1000.0 for label in labels],
        "Throughput": [results[label].throughput for label in labels],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f0f1a")
    fig.suptitle("Phase 20 Benchmark: RL vs Webster vs Static", color="#ffffff", fontsize=16)
    colors = ["#00d2ff", "#ff8c42", "#ffd93d"]

    for ax, (title, metric_values) in zip(axes.flat, values.items()):
        ax.set_facecolor("#1a1a2e")
        ax.bar(labels, metric_values, color=colors[:len(labels)])
        ax.set_title(title, color="#ffffff")
        ax.tick_params(axis="x", labelrotation=12, colors="#c0c0d0")
        ax.tick_params(axis="y", colors="#c0c0d0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a6e")
        for idx, value in enumerate(metric_values):
            ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom", color="#ffffff", fontsize=9)

    plt.tight_layout()
    out = PLOT_DIR / "phase20_benchmark_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  plot: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 20 benchmark comparison")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--base-port", type=int, default=8920)
    args = parser.parse_args()

    strategies = [
        ("rl", "PPO RL Multi-Agent"),
        ("webster", "Webster Fixed-Cycle"),
        ("static", "Static Timing"),
    ]

    results: Dict[str, BenchmarkMetrics] = {}
    print("Phase 20 benchmark starting")
    print(f"steps={args.steps} warmup={args.warmup} seed={args.seed}")
    for index, (strategy, label) in enumerate(strategies):
        print(f"\nRunning {label}")
        results[label] = run_strategy(
            strategy=strategy,
            label=label,
            steps=args.steps,
            warmup_steps=args.warmup,
            port=args.base_port + index,
            seed=args.seed,
        )

    summary = {
        "config": {
            "steps": args.steps,
            "warmup_steps": args.warmup,
            "seed": args.seed,
            "sumocfg": str(SUMOCFG),
            "model_path": str(MODEL_PATH),
        },
        "results": {label: metrics.to_dict() for label, metrics in results.items()},
        "improvements_vs_rl": improvement_table(results),
    }

    out_json = RESULT_DIR / "phase20_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nresults: {out_json}")
    save_plots(results)

    print("\nSummary")
    for label, metrics in results.items():
        print(
            f"  {label:20s} "
            f"wait={metrics.mean_waiting_s:8.2f}s "
            f"queue={metrics.mean_queue:6.2f} "
            f"co2={metrics.mean_co2_mg_s / 1000.0:8.2f}g/s "
            f"throughput={metrics.throughput:4d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
