"""
Webster Fixed-Cycle Baseline — ai-engine/webster_baseline.py
=============================================================
Implements the Webster (1958) optimal cycle formula:
    C0 = (1.5L + 5) / (1 - Y)

where:
    L  = total lost time per cycle (s)
    Y  = sum of critical-lane volume ratios (y_i = q_i / s_i)

Runs SUMO with fixed Webster-timed signals and records:
    - Waiting time
    - Queue length
    - CO2 Emissions
    - Throughput

Usage:
    python webster_baseline.py                   # run baseline only
    python webster_baseline.py --compare         # baseline + PPO comparison
    python webster_baseline.py --steps 1000      # custom episode length
"""

import os, sys, json, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SUMO / TraCI
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for _p in ["/usr/share/sumo/tools", "/opt/sumo/tools"]:
        if os.path.isdir(_p): sys.path.append(_p); break

try:
    import traci
    import traci.constants as tc
except ImportError as e:
    sys.exit(f"[ERROR] TraCI not found: {e}")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
SUMOCFG     = BASE_DIR / ".." / "sumo" / "dhanbad.sumocfg"
OUTPUT_DIR  = BASE_DIR / "output"
RESULT_DIR  = OUTPUT_DIR / "results"
PLOT_DIR    = OUTPUT_DIR / "plots"
for d in [RESULT_DIR, PLOT_DIR]: d.mkdir(parents=True, exist_ok=True)

# ── Webster parameters ────────────────────────────────────────────────────────
SATURATION_FLOW = 1800   # vehicles/hour/lane (standard value)
LOST_TIME_PER_PHASE = 4  # seconds (yellow + all-red)
MIN_CYCLE  = 30
MAX_CYCLE  = 120
DEFAULT_SPLIT = 0.5      # 50/50 green split between phases


# ═══════════════════════════════════════════════════════════════════════════════
# Webster Calculator
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class WebsterResult:
    tls_id:       str
    cycle_time:   float       # C0 (seconds)
    lost_time:    float       # L  (seconds)
    flow_ratio:   float       # Y
    phase_greens: List[float] = field(default_factory=list)


def calculate_webster(
    tls_id:       str,
    flow_rates:   List[float],   # vehicles/hour per critical lane per phase
    n_phases:     int,
    sat_flow:     float = SATURATION_FLOW,
    lost_per_ph:  float = LOST_TIME_PER_PHASE,
) -> WebsterResult:
    """
    Apply Webster's formula to compute optimal cycle and green splits.
        C0 = (1.5L + 5) / (1 - Y)
        g_i = (C0 - L) * y_i / Y
    """
    L = n_phases * lost_per_ph
    y = [q / sat_flow for q in flow_rates]   # flow ratios
    Y = sum(y)

    if Y >= 1.0:
        Y = 0.9    # saturated — cap to avoid division by zero

    C0 = (1.5 * L + 5) / (1 - Y)
    C0 = float(np.clip(C0, MIN_CYCLE, MAX_CYCLE))

    effective_green = C0 - L
    phase_greens = [(yi / Y) * effective_green if Y > 0 else effective_green / n_phases
                    for yi in y]

    return WebsterResult(
        tls_id      = tls_id,
        cycle_time  = round(C0, 1),
        lost_time   = L,
        flow_ratio  = round(Y, 4),
        phase_greens = [round(g, 1) for g in phase_greens],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline runner
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class EpisodeMetrics:
    label:         str
    total_waiting: List[float] = field(default_factory=list)
    total_queue:   List[float] = field(default_factory=list)
    total_co2:     List[float] = field(default_factory=list)
    throughput:    int         = 0
    vehicles_seen: set         = field(default_factory=set)

    @property
    def mean_wait(self):   return float(np.mean(self.total_waiting)) if self.total_waiting else 0.0
    @property
    def mean_queue(self):  return float(np.mean(self.total_queue))   if self.total_queue   else 0.0
    @property
    def mean_co2(self):    return float(np.mean(self.total_co2))     if self.total_co2     else 0.0


class WebsterBaseline:
    """Run a full SUMO episode with Webster-timed fixed signals."""

    def __init__(
        self,
        sumocfg:    Path = SUMOCFG,
        max_steps:  int  = 3600,
        port:       int  = 8840,
        label:      str  = "webster",
    ):
        self.sumocfg   = sumocfg
        self.max_steps = max_steps
        self.port      = port
        self.label     = label
        self._running  = False

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def _start(self):
        cmd = [
            "sumo", "-c", str(self.sumocfg),
            "--step-length", "1",
            "--no-warnings", "true",
            "--collision.action", "warn",
        ]
        traci.start(cmd, port=self.port, label=self.label)
        traci.switch(self.label)
        self._running = True

    def _stop(self):
        if self._running:
            try:
                traci.switch(self.label)
                traci.close()
            except Exception:
                pass
            self._running = False

    # ── Webster setup ─────────────────────────────────────────────────────────
    def _apply_webster_timing(self, warm_up_steps: int = 300):
        """
        Run warm-up steps to measure flow, then set Webster-optimal timings.
        """
        # Warm-up: collect flow counts per TLS
        flow_counts: Dict[str, List[int]] = {}
        tls_ids = list(traci.trafficlight.getIDList())

        for tls_id in tls_ids:
            logic  = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            flow_counts[tls_id] = [0] * len(logic.phases)

        print(f"  [Webster] Warm-up for {warm_up_steps} steps …")
        for _ in range(warm_up_steps):
            traci.simulationStep()
            for tls_id in tls_ids:
                ph = traci.trafficlight.getPhase(tls_id)
                ctrl = traci.trafficlight.getControlledLinks(tls_id)
                veh_count = 0
                for group in ctrl:
                    for link in group:
                        if link:
                            try:
                                veh_count += traci.lane.getLastStepVehicleNumber(link[0])
                            except Exception:
                                pass
                try:
                    flow_counts[tls_id][ph] += veh_count
                except IndexError:
                    pass

        # Compute Webster timing per TLS
        print(f"  [Webster] Computing optimal cycle times …")
        results = {}
        for tls_id in tls_ids:
            logic    = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(logic.phases)
            if n_phases == 0:
                continue

            flows = [max(1, flow_counts[tls_id][i] * 3600 / warm_up_steps)
                     for i in range(n_phases)]
            result = calculate_webster(tls_id, flows, n_phases)
            results[tls_id] = result

            # Build new program with Webster green durations
            new_phases = []
            state_list = [p.state for p in logic.phases]
            for i, ph_state in enumerate(state_list):
                green = max(5.0, result.phase_greens[i] if i < len(result.phase_greens) else 10.0)
                new_phases.append(traci.trafficlight.Phase(green, ph_state))

            try:
                new_logic = traci.trafficlight.Logic(
                    programID  = "webster",
                    type       = 0,
                    currentPhaseIndex = 0,
                    phases     = new_phases,
                )
                traci.trafficlight.setProgramLogic(tls_id, new_logic)
                traci.trafficlight.setProgram(tls_id, "webster")
                print(
                    f"    TLS {tls_id[:20]:20s} → C0={result.cycle_time:5.1f}s  "
                    f"Y={result.flow_ratio:.3f}  "
                    f"greens={result.phase_greens}"
                )
            except Exception as exc:
                print(f"    [!] Could not set program for {tls_id}: {exc}")

        return results, warm_up_steps

    # ── main run ──────────────────────────────────────────────────────────────
    def run(self, warm_up: int = 300) -> EpisodeMetrics:
        self._start()
        traci.switch(self.label)

        metrics = EpisodeMetrics(label="Webster Fixed-Cycle")
        webster_results, used_steps = self._apply_webster_timing(warm_up)
        remaining = self.max_steps - used_steps

        # Subscribe to all lanes
        all_lanes = list(traci.lane.getIDList())
        lane_vars  = [
            tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
            tc.LAST_STEP_VEHICLE_NUMBER,
            tc.VAR_WAITING_TIME,
        ]
        for lid in all_lanes:
            traci.lane.subscribe(lid, lane_vars)

        print(f"  [Webster] Running {remaining} measurement steps …")
        for step in range(remaining):
            traci.simulationStep()
            lane_res = traci.lane.getAllSubscriptionResults()

            step_queue = sum(v.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
                             for v in lane_res.values())
            step_wait  = sum(v.get(tc.VAR_WAITING_TIME, 0.0)
                             for v in lane_res.values())
            step_co2   = sum(
                traci.vehicle.getCO2Emission(vid)
                for vid in traci.vehicle.getIDList()
                if vid not in metrics.vehicles_seen
            ) + sum(
                traci.vehicle.getCO2Emission(vid)
                for vid in traci.vehicle.getIDList()
                if vid in metrics.vehicles_seen
            )
            # Track throughput
            for vid in traci.vehicle.getIDList():
                metrics.vehicles_seen.add(vid)

            metrics.total_queue.append(step_queue)
            metrics.total_waiting.append(step_wait)
            metrics.total_co2.append(step_co2)

            if step % 200 == 0:
                print(
                    f"    Step {step:4d}/{remaining} | "
                    f"queue={step_queue:4d} | "
                    f"wait={step_wait:8.1f}s | "
                    f"CO2={step_co2:6.0f}mg/s"
                )

        metrics.throughput = len(metrics.vehicles_seen)
        self._stop()
        return metrics, webster_results


# ═══════════════════════════════════════════════════════════════════════════════
# PPO comparison runner
# ═══════════════════════════════════════════════════════════════════════════════
def run_ppo_comparison(max_steps: int = 3600) -> EpisodeMetrics:
    """Load saved PPO model and run an episode to collect metrics."""
    model_path = OUTPUT_DIR / "models" / "ppo_traffic_model.zip"
    if not model_path.exists():
        print(f"  [!] PPO model not found at {model_path} — skipping comparison")
        return None

    from stable_baselines3 import PPO
    sys.path.insert(0, str(BASE_DIR))
    from environment import SmartCityTrafficEnv

    print("\n  [PPO] Loading model …")
    env   = SmartCityTrafficEnv(use_gui=False, max_steps=max_steps, port=8845, seed=999)
    model = PPO.load(str(model_path), env=env)

    metrics = EpisodeMetrics(label="PPO RL Agent")
    obs, _  = env.reset()

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))

        metrics.total_queue.append(info.get("total_queue",   0))
        metrics.total_waiting.append(info.get("total_waiting", 0.0))
        metrics.total_co2.append(0.0)   # env doesn't expose CO2 directly

        for vid in traci.vehicle.getIDList() if env._sumo_running else []:
            metrics.vehicles_seen.add(vid)

        if terminated or truncated:
            break

    metrics.throughput = len(metrics.vehicles_seen)
    env.close()
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison plots
# ═══════════════════════════════════════════════════════════════════════════════
def plot_comparison(results: Dict[str, EpisodeMetrics], save_path: Path):
    labels  = list(results.keys())
    colors  = ["#ff6b6b", "#00d2ff", "#ffd93d", "#6bcb77"][:len(labels)]

    metrics_def = [
        ("Mean Wait (s)",       [r.mean_wait  for r in results.values()]),
        ("Mean Queue (veh)",    [r.mean_queue for r in results.values()]),
        ("Mean CO₂ (mg/s)",     [r.mean_co2   for r in results.values()]),
        ("Throughput (veh)",    [r.throughput  for r in results.values()]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f0f1a")
    fig.suptitle("AI vs Webster Fixed-Cycle Comparison", color="#ffffff",
                 fontsize=16, fontweight="bold", y=1.01)

    for ax, (title, values) in zip(axes.flat, metrics_def):
        ax.set_facecolor("#1a1a2e")
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#0f0f1a")
        ax.set_title(title, color="#c0c0d0", fontsize=12)
        ax.tick_params(colors="#c0c0d0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a6e")
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values, default=1),
                    f"{val:.1f}", ha="center", va="bottom", color="#ffffff", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Comparison plot → {save_path}")


def plot_timeseries(results: Dict[str, EpisodeMetrics], save_path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor="#0f0f1a")
    palette = {"Webster Fixed-Cycle": "#ff6b6b", "PPO RL Agent": "#00d2ff"}

    for label, m in results.items():
        color = palette.get(label, "#ffd93d")
        steps = np.arange(len(m.total_queue))
        ax1.plot(steps, m.total_queue,   color=color, alpha=0.7, linewidth=1.2, label=label)
        ax2.plot(steps, m.total_waiting, color=color, alpha=0.7, linewidth=1.2, label=label)

    for ax, title, ylabel in [
        (ax1, "Queue Length Over Time",  "Vehicles halted"),
        (ax2, "Total Waiting Time",      "Cumulative wait (s)"),
    ]:
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, color="#c0c0d0")
        ax.tick_params(colors="#c0c0d0")
        for spine in ax.spines.values(): spine.set_edgecolor("#3a3a6e")
        ax.legend(facecolor="#1a1a2e", edgecolor="#3a3a6e", labelcolor="#c0c0d0")
        ax.grid(True, color="#2a2a4e", linewidth=0.5)

    ax2.set_xlabel("Simulation Step (s)", color="#c0c0d0")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Time-series plot → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webster baseline comparison")
    parser.add_argument("--steps",   type=int, default=1000, help="Episode steps (default: 1000)")
    parser.add_argument("--warmup",  type=int, default=200,  help="Warm-up steps for flow measurement")
    parser.add_argument("--compare", action="store_true",    help="Also run PPO agent for comparison")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Phase 7 — Webster Fixed-Cycle Baseline              ║")
    print(f"║  Steps: {args.steps}  Warm-up: {args.warmup}  Compare PPO: {args.compare}")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Run Webster baseline ─────────────────────────────────────────────────
    runner  = WebsterBaseline(max_steps=args.steps, port=8840)
    w_metrics, w_results = runner.run(warm_up=args.warmup)

    print(f"\n  ┌── Webster Baseline Results ────────────────────────")
    print(f"  │  Mean queue   : {w_metrics.mean_queue:.2f} vehicles")
    print(f"  │  Mean wait    : {w_metrics.mean_wait:.2f} s")
    print(f"  │  Mean CO₂     : {w_metrics.mean_co2:.0f} mg/s")
    print(f"  │  Throughput   : {w_metrics.throughput} vehicles")
    print(f"  └────────────────────────────────────────────────────")

    # Print Webster parameters table
    print(f"\n  Webster Parameters:")
    print(f"  {'TLS ID':25s} {'C0 (s)':8s} {'Y':8s} {'Greens'}")
    print(f"  {'-'*60}")
    for tls_id, res in w_results.items():
        print(f"  {tls_id[:25]:25s} {res.cycle_time:8.1f} {res.flow_ratio:8.4f} {res.phase_greens}")

    all_results = {"Webster Fixed-Cycle": w_metrics}

    # ── Optional PPO comparison ───────────────────────────────────────────────
    if args.compare:
        ppo_metrics = run_ppo_comparison(max_steps=args.steps)
        if ppo_metrics:
            all_results["PPO RL Agent"] = ppo_metrics
            print(f"\n  ┌── PPO Agent Results ───────────────────────────────")
            print(f"  │  Mean queue   : {ppo_metrics.mean_queue:.2f} vehicles")
            print(f"  │  Mean wait    : {ppo_metrics.mean_wait:.2f} s")
            print(f"  │  Throughput   : {ppo_metrics.throughput} vehicles")
            print(f"  └────────────────────────────────────────────────────")

            # Improvement %
            if w_metrics.mean_wait > 0:
                wait_imp = 100*(w_metrics.mean_wait - ppo_metrics.mean_wait)/w_metrics.mean_wait
                print(f"\n  Waiting time improvement : {wait_imp:+.1f}%")
            if w_metrics.mean_queue > 0:
                queue_imp = 100*(w_metrics.mean_queue - ppo_metrics.mean_queue)/w_metrics.mean_queue
                print(f"  Queue length improvement : {queue_imp:+.1f}%")

    # ── Save results JSON ─────────────────────────────────────────────────────
    summary = {
        label: {
            "mean_wait":   m.mean_wait,
            "mean_queue":  m.mean_queue,
            "mean_co2":    m.mean_co2,
            "throughput":  m.throughput,
        }
        for label, m in all_results.items()
    }
    out_json = RESULT_DIR / "webster_comparison.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Results saved → {out_json}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots …")
    plot_comparison(all_results, PLOT_DIR / "comparison_bar.png")
    plot_timeseries(all_results, PLOT_DIR / "comparison_timeseries.png")

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Done! Run with --compare to include PPO results.    ║")
    print("╚══════════════════════════════════════════════════════╝")
