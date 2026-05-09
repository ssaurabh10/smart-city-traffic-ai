"""
PPO Training Script — Smart City Traffic AI
============================================
Trains a PPO agent using Stable-Baselines3 to control traffic signals
in the Dhanbad SUMO simulation.

Usage:
  python train.py                          # default 100k timesteps
  python train.py --timesteps 500000       # longer training
  python train.py --timesteps 50000 --eval # quick run with evaluation
  python train.py --resume models/ppo_traffic_model.zip  # continue training

Outputs (in ai-engine/output/):
  models/ppo_traffic_model.zip    — final trained model
  models/best_model.zip           — best checkpoint during training
  logs/ppo_traffic/               — TensorBoard logs
  plots/reward_curve.png          — episode reward over time
  plots/metrics.png               — queue / waiting / speed plots
  results/eval_metrics.json       — final evaluation stats
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── SB3 imports ───────────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from environment import SmartCityTrafficEnv, RewardConfig

# ── Output directory layout ───────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
MODEL_DIR   = OUTPUT_DIR / "models"
LOG_DIR     = OUTPUT_DIR / "logs"
PLOT_DIR    = OUTPUT_DIR / "plots"
RESULT_DIR  = OUTPUT_DIR / "results"

for _d in [MODEL_DIR, LOG_DIR, PLOT_DIR, RESULT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

class TrafficMetricsCallback(BaseCallback):
    """
    Logs traffic-specific KPIs (queue, waiting, speed) to TensorBoard
    and stores them for offline plotting.
    """

    def __init__(self, log_interval: int = 500, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval    = log_interval
        self.episode_rewards: list = []
        self.episode_queues:  list = []
        self.episode_waits:   list = []
        self._ep_reward      = 0.0
        self._ep_queue:  list = []
        self._ep_wait:   list = []

    def _on_step(self) -> bool:
        # Accumulate within episode
        info           = self.locals["infos"][0]
        reward         = self.locals["rewards"][0]
        self._ep_reward += reward
        self._ep_queue.append(info.get("total_queue",   0))
        self._ep_wait.append( info.get("total_waiting", 0.0))

        # Log to TensorBoard every N steps
        if self.n_calls % self.log_interval == 0:
            self.logger.record("traffic/queue_mean",
                               np.mean(self._ep_queue[-100:]))
            self.logger.record("traffic/wait_mean",
                               np.mean(self._ep_wait[-100:]))
            self.logger.record("traffic/step_reward", reward)

        # Episode boundary
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_queues.append(np.mean(self._ep_queue))
            self.episode_waits.append(np.mean(self._ep_wait))

            self.logger.record("traffic/episode_reward", self._ep_reward)
            self.logger.record("traffic/episode_queue_mean",
                               np.mean(self._ep_queue))
            self.logger.record("traffic/episode_wait_mean",
                               np.mean(self._ep_wait))

            if self.verbose >= 1 and len(self.episode_rewards) % 5 == 0:
                ep = len(self.episode_rewards)
                print(
                    f"  [Episode {ep:3d}] "
                    f"reward={self._ep_reward:+7.2f} | "
                    f"queue={np.mean(self._ep_queue):5.1f} | "
                    f"wait={np.mean(self._ep_wait):7.1f}s"
                )

            # Reset accumulators
            self._ep_reward = 0.0
            self._ep_queue  = []
            self._ep_wait   = []

        return True   # True = continue training


class ProgressCallback(BaseCallback):
    """Prints a compact progress line every N steps."""

    def __init__(self, total_steps: int, print_every: int = 5000):
        super().__init__()
        self.total_steps = total_steps
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.n_calls % self.print_every == 0:
            pct = 100 * self.n_calls / self.total_steps
            bar_len = 30
            filled  = int(bar_len * self.n_calls / self.total_steps)
            bar     = "█" * filled + "░" * (bar_len - filled)
            print(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"step {self.n_calls:>7,}/{self.total_steps:,}",
                end="", flush=True
            )
            if self.n_calls == self.total_steps:
                print()   # newline at end
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Environment factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(
    port:      int  = 8820,
    max_steps: int  = 3600,
    seed:      int  = 42,
    monitor_dir: str | None = None,
):
    """Return a (optionally monitored) SmartCityTrafficEnv factory."""
    def _factory():
        env = SmartCityTrafficEnv(
            use_gui    = False,
            max_steps  = max_steps,
            yellow_dur = 4,
            min_green  = 10,
            reward_cfg = RewardConfig(
                alpha=0.3, beta=0.1, gamma=0.5, delta=1.0
            ),
            port = port,
            seed = seed,
        )
        if monitor_dir:
            env = Monitor(env, monitor_dir)
        return env
    return _factory


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _smooth(values: list, window: int = 10) -> np.ndarray:
    """Simple moving average."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_reward_curve(
    metrics_cb: TrafficMetricsCallback,
    save_path:  Path,
):
    """Plot episode reward over training."""
    rewards  = metrics_cb.episode_rewards
    if not rewards:
        return

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    episodes = np.arange(1, len(rewards) + 1)
    # Raw
    ax.plot(episodes, rewards, color="#3a3a6e", alpha=0.4,
            linewidth=0.8, label="raw")
    # Smoothed
    if len(rewards) >= 10:
        smooth = _smooth(rewards, 10)
        ep_s   = episodes[9:]
        ax.plot(ep_s, smooth, color="#00d2ff", linewidth=2.0,
                label="smoothed (10-ep)")
        # Shade under curve
        ax.fill_between(ep_s, smooth, alpha=0.15, color="#00d2ff")

    ax.axhline(0, color="#ffffff", alpha=0.2, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Episode", color="#c0c0d0", fontsize=12)
    ax.set_ylabel("Total Reward", color="#c0c0d0", fontsize=12)
    ax.set_title("PPO Training — Episode Reward", color="#ffffff",
                 fontsize=14, fontweight="bold")
    ax.tick_params(colors="#c0c0d0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a6e")
    ax.legend(facecolor="#1a1a2e", edgecolor="#3a3a6e", labelcolor="#c0c0d0")
    ax.grid(True, color="#2a2a4e", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Reward curve saved → {save_path}")


def plot_traffic_metrics(
    metrics_cb: TrafficMetricsCallback,
    save_path:  Path,
):
    """Plot queue and waiting time over episodes."""
    queues = metrics_cb.episode_queues
    waits  = metrics_cb.episode_waits
    if not queues:
        return

    fig = plt.figure(figsize=(14, 8), facecolor="#0f0f1a")
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4)
    eps = np.arange(1, len(queues) + 1)

    palette = {"queue": "#ff6b6b", "wait": "#ffd93d"}

    for ax, data, label, color in [
        (fig.add_subplot(gs[0]), queues, "Avg Queue Length (vehicles)", palette["queue"]),
        (fig.add_subplot(gs[1]), waits,  "Avg Wait Time (s)",           palette["wait"]),
    ]:
        ax.set_facecolor("#1a1a2e")
        ax.plot(eps, data, color=color, alpha=0.35, linewidth=0.8)
        if len(data) >= 10:
            smooth = _smooth(data, 10)
            ax.plot(eps[9:], smooth, color=color, linewidth=2.0)
            ax.fill_between(eps[9:], smooth, alpha=0.15, color=color)
        ax.set_ylabel(label, color="#c0c0d0", fontsize=11)
        ax.tick_params(colors="#c0c0d0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a6e")
        ax.grid(True, color="#2a2a4e", linewidth=0.5)

    fig.axes[0].set_title("PPO Training — Traffic KPIs", color="#ffffff",
                           fontsize=14, fontweight="bold")
    fig.axes[-1].set_xlabel("Episode", color="#c0c0d0", fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Metrics plot saved  → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    n_episodes:  int  = 5,
    max_steps:   int  = 1000,
    eval_port:   int  = 8830,
) -> dict:
    """
    Run the trained model for n_episodes and return aggregated KPIs.
    """
    print(f"\n{'─'*60}")
    print(f"  Evaluating over {n_episodes} episodes …")

    env = SmartCityTrafficEnv(
        use_gui   = False,
        max_steps = max_steps,
        port      = eval_port,
        seed      = 999,
    )

    ep_rewards, ep_queues, ep_waits = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_r, queues, waits = 0.0, [], []

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_r   += reward
            queues.append(info.get("total_queue",   0))
            waits.append( info.get("total_waiting", 0.0))
            if terminated or truncated:
                break

        ep_rewards.append(ep_r)
        ep_queues.append(np.mean(queues))
        ep_waits.append(np.mean(waits))
        print(
            f"    Ep {ep+1}/{n_episodes}: "
            f"reward={ep_r:+7.2f} | "
            f"avg_queue={np.mean(queues):.1f} | "
            f"avg_wait={np.mean(waits):.1f}s"
        )

    env.close()

    results = {
        "episodes":           n_episodes,
        "mean_reward":        float(np.mean(ep_rewards)),
        "std_reward":         float(np.std(ep_rewards)),
        "mean_queue":         float(np.mean(ep_queues)),
        "mean_wait_s":        float(np.mean(ep_waits)),
        "per_episode_rewards": [float(r) for r in ep_rewards],
        "per_episode_queues":  [float(q) for q in ep_queues],
        "per_episode_waits":   [float(w) for w in ep_waits],
    }

    print(f"\n  ┌── Evaluation Summary ──────────────────────────")
    print(f"  │  Mean reward  : {results['mean_reward']:+.2f} ± {results['std_reward']:.2f}")
    print(f"  │  Mean queue   : {results['mean_queue']:.2f} vehicles")
    print(f"  │  Mean wait    : {results['mean_wait_s']:.1f} s")
    print(f"  └────────────────────────────────────────────────")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    total_timesteps: int  = 100_000,
    resume_path:     str  = None,
    run_eval:        bool = False,
    episode_steps:   int  = 3600,
    seed:            int  = 42,
):
    print("╔══════════════════════════════════════════════════════╗")
    print("║     Smart City Traffic AI — PPO Training             ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Timesteps   : {total_timesteps:,}")
    print(f"║  Episode len : {episode_steps} steps")
    print(f"║  Seed        : {seed}")
    print(f"║  Output dir  : {OUTPUT_DIR}")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Build training environment ────────────────────────────────────────────
    monitor_dir = str(LOG_DIR / "monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    train_env = DummyVecEnv([make_env(
        port      = 8820,
        max_steps = episode_steps,
        seed      = seed,
        monitor_dir = monitor_dir,
    )])

    # ── Sanity check the raw (unwrapped) env ─────────────────────────────────
    print("  Running environment sanity check …")
    raw_env = SmartCityTrafficEnv(
        use_gui   = False,
        max_steps = 20,
        port      = 8825,
        seed      = seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(raw_env, warn=True)
    raw_env.close()
    print("  ✓ Environment check passed\n")

    # ── PPO hyper-parameters ──────────────────────────────────────────────────
    ppo_kwargs = dict(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 2048,          # rollout buffer size
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,          # discount factor
        gae_lambda      = 0.95,          # GAE lambda
        clip_range      = 0.2,
        ent_coef        = 0.01,          # entropy bonus (encourages exploration)
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        tensorboard_log = str(LOG_DIR),
        verbose         = 0,
        seed            = seed,
        policy_kwargs   = dict(
            net_arch = [dict(pi=[256, 256], vf=[256, 256])],
        ),
    )

    # ── Create or resume model ────────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        print(f"  Resuming from: {resume_path}")
        model = PPO.load(
            resume_path,
            env          = train_env,
            tensorboard_log = str(LOG_DIR),
        )
    else:
        print("  Creating new PPO model …")
        model = PPO(**ppo_kwargs)

    print(f"  Policy architecture: {model.policy}")
    print(f"  Total parameters   : "
          f"{sum(p.numel() for p in model.policy.parameters()):,}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    metrics_cb = TrafficMetricsCallback(log_interval=500, verbose=1)
    progress_cb = ProgressCallback(
        total_steps = total_timesteps,
        print_every = max(1000, total_timesteps // 50),
    )
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(10_000, total_timesteps // 10),
        save_path   = str(MODEL_DIR / "checkpoints"),
        name_prefix = "ppo_traffic",
        verbose     = 0,
    )
    eval_env_cb = DummyVecEnv([make_env(port=8821, max_steps=500, seed=seed+1)])
    eval_cb = EvalCallback(
        eval_env         = eval_env_cb,
        best_model_save_path = str(MODEL_DIR),
        log_path         = str(LOG_DIR / "eval"),
        eval_freq        = max(5_000, total_timesteps // 20),
        n_eval_episodes  = 3,
        deterministic    = True,
        render           = False,
        verbose          = 0,
    )

    callback = CallbackList([metrics_cb, progress_cb, checkpoint_cb, eval_cb])

    # ── Train ─────────────────────────────────────────────────────────────────
    print("  Training started …\n")
    try:
        model.learn(
            total_timesteps = total_timesteps,
            callback        = callback,
            tb_log_name     = "ppo_traffic",
            reset_num_timesteps = (resume_path is None),
            progress_bar    = False,
        )
    except KeyboardInterrupt:
        print("\n\n  [!] Training interrupted by user — saving checkpoint …")
    finally:
        print()

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = MODEL_DIR / "ppo_traffic_model"
    model.save(str(final_path))
    print(f"\n  ✓ Final model saved → {final_path}.zip")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\n  Generating plots …")
    plot_reward_curve(metrics_cb,   PLOT_DIR / "reward_curve.png")
    plot_traffic_metrics(metrics_cb, PLOT_DIR / "metrics.png")

    # ── Optional evaluation ───────────────────────────────────────────────────
    if run_eval:
        results = evaluate_model(
            model,
            n_episodes  = 5,
            max_steps   = min(500, episode_steps),
            eval_port   = 8830,
        )
        result_path = RESULT_DIR / "eval_metrics.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ Eval metrics saved → {result_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    train_env.close()
    eval_env_cb.close()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Training complete!                                  ║")
    print(f"║  Model   : {str(final_path)}.zip")
    print(f"║  Plots   : {PLOT_DIR}")
    print(f"║  Logs    : {LOG_DIR}")
    print("║                                                      ║")
    print("║  To view TensorBoard:                                ║")
    print(f"║    tensorboard --logdir {LOG_DIR}")
    print("╚══════════════════════════════════════════════════════╝\n")

    return model, metrics_cb


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO traffic signal controller"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps (default: 100,000)"
    )
    parser.add_argument(
        "--episode-steps", type=int, default=3600,
        help="Steps per episode / sim seconds (default: 3600)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a .zip model to resume training from"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation after training and save metrics"
    )
    args = parser.parse_args()

    train(
        total_timesteps = args.timesteps,
        resume_path     = args.resume,
        run_eval        = args.eval,
        episode_steps   = args.episode_steps,
        seed            = args.seed,
    )
