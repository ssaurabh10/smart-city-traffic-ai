"""
SmartCityTrafficEnv — PPO Reinforcement Learning Environment
=============================================================
A Gymnasium-compatible environment that wraps the SUMO simulation
via TraCI. One agent controls one TLS junction at a time; multiple
agents can be composed for multi-intersection control.

State Space (per intersection):
  • Queue lengths       — vehicles halted per approach lane
  • Signal phase        — one-hot current phase + time since switch
  • Vehicle density     — vehicles / lane capacity per lane
  • Bus distance        — proximity of nearest bus to each approach
  • Emergency proximity — nearest emergency vehicle distance

Action Space (Discrete 4):
  0 — Keep current phase (extend green)
  1 — Switch to next phase
  2 — Skip one phase (jump to phase+2)
  3 — Emergency override (force all-red → priority green)

Reward Function:
  R_t = -Δwait - α·queue - β·CO₂ + γ·bus_priority + δ·emergency_priority

Tunable coefficients α, β, γ, δ via RewardConfig.
"""

import os
import sys
import math
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── SUMO / TraCI path setup ──────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for _p in ["/usr/share/sumo/tools", "/opt/sumo/tools"]:
        if os.path.isdir(_p):
            sys.path.append(_p)
            break

try:
    import traci
    import traci.constants as tc
except ImportError as e:
    sys.exit(f"[ERROR] TraCI not found: {e}\n"
             "Set SUMO_HOME or install Eclipse SUMO.")

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "..", "sumo", "dhanbad.sumocfg")
SUMO_PORT    = 8814                      # different port from manual controller

# Vehicle type IDs used in the .rou.xml (adjust to match yours)
BUS_TYPES       = {"bus", "BUS", "pt_bus"}
EMERGENCY_TYPES = {"emergency", "ambulance", "police", "fire"}

# ── Reward hyperparameters ────────────────────────────────────────────────────
@dataclass
class RewardConfig:
    """
    Coefficients for:
      R_t = -Δwait - α·queue - β·CO₂ + γ·bus_priority + δ·emergency_priority
    """
    alpha:     float = 0.3    # queue penalty weight
    beta:      float = 0.1    # CO₂ / emissions penalty weight
    gamma:     float = 0.5    # bus priority bonus weight
    delta:     float = 1.0    # emergency vehicle priority bonus weight
    wait_norm: float = 200.0  # normalisation constant for waiting time (s)
    queue_norm: float = 20.0  # normalisation constant for queue (vehicles)
    co2_norm:  float = 5000.0 # normalisation constant for CO₂ (mg/s)
    bus_dist_threshold:       float = 150.0  # metres — "near" bus
    emergency_dist_threshold: float = 300.0  # metres — "near" emergency veh


# ── State / Action dimensions ─────────────────────────────────────────────────
MAX_LANES      = 8    # max controlled approach lanes per intersection
MAX_PHASES     = 8    # max TLS phases
N_LANE_FEATS   = 5    # [queue, waiting, density, speed, occupancy] per lane
EXTRA_FEATS    = 3    # [phase_onehot×MAX_PHASES packed as idx, phase_time,
                      #  emergency_flag]  → see _build_state for detail
# Total obs dimension:
OBS_DIM = MAX_LANES * N_LANE_FEATS + MAX_PHASES + 2   # +2: phase_time, emerg


# ── Gymnasium Environment ─────────────────────────────────────────────────────
class SmartCityTrafficEnv(gym.Env):
    """
    Single-intersection PPO environment.

    Parameters
    ----------
    tls_id      : SUMO traffic light ID to control
    sumocfg     : path to .sumocfg file
    use_gui     : launch sumo-gui instead of headless sumo
    max_steps   : episode length in simulation steps (seconds)
    yellow_dur  : mandatory yellow phase duration inserted on every switch (s)
    min_green   : minimum green duration before a switch is allowed (s)
    reward_cfg  : RewardConfig instance
    port        : TraCI TCP port
    seed        : random seed (passed to SUMO)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        tls_id:     str           = "",
        sumocfg:    str           = SUMOCFG_PATH,
        use_gui:    bool          = False,
        max_steps:  int           = 3600,
        yellow_dur: int           = 4,
        min_green:  int           = 10,
        reward_cfg: RewardConfig  = None,
        port:       int           = SUMO_PORT,
        seed:       int           = 42,
    ):
        super().__init__()
        self.tls_id      = tls_id
        self.sumocfg     = os.path.abspath(sumocfg)
        self.use_gui     = use_gui
        self.max_steps   = max_steps
        self.yellow_dur  = yellow_dur
        self.min_green   = min_green
        self.rcfg        = reward_cfg or RewardConfig()
        self.port        = port
        self.seed_val    = seed
        self._label      = f"sumo_{port}"   # unique TraCI connection label

        # Internal state
        self._step:            int   = 0
        self._phase:           int   = 0
        self._phase_timer:     int   = 0          # seconds in current phase
        self._in_yellow:       bool  = False
        self._yellow_counter:  int   = 0
        self._prev_waiting:    float = 0.0
        self._sumo_running:    bool  = False

        # Will be filled after SUMO starts
        self._n_phases:        int   = 0
        self._ctrl_lanes:      List[str] = []
        self._lane_caps:       Dict[str, float] = {}  # lane_id → capacity

        # Spaces
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (OBS_DIM,),
            dtype = np.float32,
        )
        # 0=keep, 1=next, 2=skip, 3=emergency_override
        self.action_space = spaces.Discrete(4)

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        traci.switch(self._label) if self._sumo_running else None

        self._stop_sumo()
        self._start_sumo()

        self._step         = 0
        self._phase        = traci.trafficlight.getPhase(self.tls_id)
        self._phase_timer  = 0
        self._in_yellow    = False
        self._yellow_counter = 0
        self._prev_waiting = 0.0

        obs  = self._build_state()
        info = self._info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action, advance one simulation second, return (obs, reward, …).
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        traci.switch(self._label)   # ensure this env's connection is active

        # ── Execute action ───────────────────────────────────────────────────
        if self._in_yellow:
            # Waiting out mandatory yellow — ignore agent action
            self._yellow_counter += 1
            if self._yellow_counter >= self.yellow_dur:
                self._in_yellow = False
                self._yellow_counter = 0
                # Commit to the new phase (clamped to actual phase count)
                self._phase = self._phase % self._n_phases
                self._safe_set_phase(self._phase)
        else:
            self._apply_action(action)

        # ── Advance simulation ───────────────────────────────────────────────
        traci.simulationStep()
        self._step        += 1
        self._phase_timer += 1

        # ── Compute reward ───────────────────────────────────────────────────
        reward = self._compute_reward()

        # ── Build next observation ───────────────────────────────────────────
        obs  = self._build_state()
        info = self._info()

        terminated = (self._step >= self.max_steps)
        truncated  = (traci.simulation.getMinExpectedNumber() == 0
                      and self._step > 10)   # all vehicles left early

        return obs, reward, terminated, truncated, info

    def close(self):
        self._stop_sumo()

    def render(self):
        # GUI is launched if use_gui=True; no separate render needed
        pass

    # ── SUMO lifecycle ───────────────────────────────────────────────────────

    def _start_sumo(self):
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary,
            "-c",              self.sumocfg,
            "--step-length",   "1",
            "--seed",          str(self.seed_val),
            "--no-warnings",   "true",
            "--collision.action", "warn",
            "--random",        "false",
        ]
        traci.start(cmd, port=self.port, label=self._label)
        self._sumo_running = True
        traci.switch(self._label)

        # Auto-detect TLS if not specified — prefer most complex junction
        tls_ids = list(traci.trafficlight.getIDList())
        if not self.tls_id:
            if not tls_ids:
                raise RuntimeError("No traffic lights found in network!")
            # Pick TLS with most phases AND most controlled lanes
            def tls_score(tid):
                try:
                    logics = traci.trafficlight.getAllProgramLogics(tid)
                    n_phases = len(logics[0].phases) if logics else 0
                    n_links  = sum(
                        1 for g in traci.trafficlight.getControlledLinks(tid)
                        for lnk in g if lnk
                    )
                    return n_phases * 10 + n_links
                except Exception:
                    return 0
            self.tls_id = max(tls_ids, key=tls_score)
            print(f"[Env] Auto-selected TLS: {self.tls_id}")

        # Cache phase count and controlled lanes
        logic          = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self._n_phases = len(logic.phases)

        raw_ctrl = traci.trafficlight.getControlledLinks(self.tls_id)
        seen, ordered = set(), []
        for group in raw_ctrl:
            for link in group:
                if link and link[0] not in seen:
                    seen.add(link[0])
                    ordered.append(link[0])
        self._ctrl_lanes = ordered[:MAX_LANES]

        # Estimate lane capacities (length / 7.5 m per vehicle)
        for lid in self._ctrl_lanes:
            try:
                length = traci.lane.getLength(lid)
            except Exception:
                length = 150.0
            self._lane_caps[lid] = max(1.0, length / 7.5)

        # Subscribe for efficiency
        lane_vars = [
            tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
            tc.LAST_STEP_VEHICLE_NUMBER,
            tc.LAST_STEP_MEAN_SPEED,
            tc.LAST_STEP_OCCUPANCY,
            tc.VAR_WAITING_TIME,
        ]
        for lid in self._ctrl_lanes:
            traci.lane.subscribe(lid, lane_vars)

    def _stop_sumo(self):
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

    # ── Action execution ─────────────────────────────────────────────────────

    def _apply_action(self, action: int):
        """
        Actions:
          0 — Keep (extend green)
          1 — Switch to next phase
          2 — Skip to phase + 2
          3 — Emergency override (all-red then priority green)
        """
        if action == 0:
            # Extend green: do nothing (stay in current phase)
            return

        if action == 3:
            # Emergency override: force all-red immediately
            all_red = "r" * len(
                traci.trafficlight.getRedYellowGreenState(self.tls_id)
            )
            try:
                traci.trafficlight.setRedYellowGreenState(self.tls_id, all_red)
            except Exception:
                pass
            self._in_yellow    = True
            self._yellow_counter = 0
            # After yellow, will switch to phase that serves priority direction
            self._phase = self._priority_phase()
            return

        # Enforce minimum green
        if self._phase_timer < self.min_green:
            return   # too early — ignore switch request

        if action == 1:
            new_phase = (self._phase + 1) % self._n_phases
        else:   # action == 2
            new_phase = (self._phase + 2) % self._n_phases

        # Insert yellow transition
        try:
            logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
            cur_idx   = self._phase % len(logic.phases)
            cur_state = list(logic.phases[cur_idx].state)
            yel_state = [
                "y" if c in ("G", "g") else c
                for c in cur_state
            ]
            traci.trafficlight.setRedYellowGreenState(
                self.tls_id, "".join(yel_state)
            )
        except Exception as exc:
            print(f"[Env] Warning: yellow transition failed: {exc}")
        self._in_yellow    = True
        self._yellow_counter = 0
        self._phase        = new_phase % self._n_phases
        self._phase_timer  = 0

    def _safe_set_phase(self, phase: int):
        """Set TLS phase only if the junction has multiple phases."""
        if self._n_phases <= 1:
            return   # single-phase TLS — nothing to switch
        safe = phase % self._n_phases
        try:
            traci.trafficlight.setPhase(self.tls_id, safe)
        except Exception as exc:
            pass    # silently skip — SUMO may override during yellow

    def _priority_phase(self) -> int:
        """
        Return the TLS phase index that serves the most congested approach,
        used after an emergency override action.
        """
        lane_results = traci.lane.getAllSubscriptionResults()
        best_phase, best_queue = 0, -1
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]

        for pi, phase_obj in enumerate(logic.phases):
            state  = phase_obj.state
            q_sum  = 0
            for li, lid in enumerate(self._ctrl_lanes):
                if li < len(state) and state[li].lower() == "g":
                    vals  = lane_results.get(lid, {})
                    q_sum += vals.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            if q_sum > best_queue:
                best_queue = q_sum
                best_phase = pi
        return best_phase

    # ── State construction ───────────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """
        Construct normalised observation vector:
          [ lane_feats × MAX_LANES | phase_onehot × MAX_PHASES |
            phase_time_norm | emergency_flag ]
        """
        lane_results = traci.lane.getAllSubscriptionResults()
        obs: List[float] = []

        # ── Per-lane features ────────────────────────────────────────────────
        for lid in self._ctrl_lanes:
            vals = lane_results.get(lid, {})
            q    = vals.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            n    = vals.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            spd  = vals.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
            occ  = vals.get(tc.LAST_STEP_OCCUPANCY, 0.0)
            wait = vals.get(tc.VAR_WAITING_TIME, 0.0)
            cap  = self._lane_caps.get(lid, 20.0)

            obs.extend([
                min(q   / self.rcfg.queue_norm, 1.0),   # queue (norm)
                min(wait / self.rcfg.wait_norm,  1.0),   # waiting time (norm)
                min(n   / cap,                   1.0),   # density
                min(spd  / 14.0,                 1.0),   # speed (14 m/s ≈ 50 km/h)
                occ / 100.0,                              # occupancy
            ])

        # Pad missing lanes
        pad = MAX_LANES - len(self._ctrl_lanes)
        obs.extend([0.0] * (pad * N_LANE_FEATS))

        # ── Phase one-hot ────────────────────────────────────────────────────
        phase_oh = [0.0] * MAX_PHASES
        if 0 <= self._phase < MAX_PHASES:
            phase_oh[self._phase] = 1.0
        obs.extend(phase_oh)

        # ── Phase time (normalised to 120 s max) ────────────────────────────
        obs.append(min(self._phase_timer / 120.0, 1.0))

        # ── Emergency vehicle flag ───────────────────────────────────────────
        obs.append(1.0 if self._emergency_nearby() else 0.0)

        return np.array(obs, dtype=np.float32)

    # ── Reward computation ───────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        R_t = -Δwait - α·queue - β·CO₂ + γ·bus_priority + δ·emergency_priority
        """
        lane_results = traci.lane.getAllSubscriptionResults()
        r            = self.rcfg

        # ── Δ waiting time ────────────────────────────────────────────────────
        total_wait = sum(
            lane_results.get(lid, {}).get(tc.VAR_WAITING_TIME, 0.0)
            for lid in self._ctrl_lanes
        )
        delta_wait      = (total_wait - self._prev_waiting) / r.wait_norm
        self._prev_wait = total_wait
        self._prev_waiting = total_wait

        # ── Queue penalty ─────────────────────────────────────────────────────
        total_queue = sum(
            lane_results.get(lid, {}).get(
                tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            for lid in self._ctrl_lanes
        )
        queue_penalty = r.alpha * min(total_queue / r.queue_norm, 1.0)

        # ── CO₂ penalty ───────────────────────────────────────────────────────
        co2 = self._get_co2_emission()
        co2_penalty = r.beta * min(co2 / r.co2_norm, 1.0)

        # ── Bus priority bonus ────────────────────────────────────────────────
        bus_bonus = r.gamma * self._bus_priority_bonus()

        # ── Emergency priority bonus ──────────────────────────────────────────
        emerg_bonus = r.delta * self._emergency_priority_bonus()

        reward = (
            -delta_wait
            - queue_penalty
            - co2_penalty
            + bus_bonus
            + emerg_bonus
        )
        return float(reward)

    # ── Auxiliary metric helpers ─────────────────────────────────────────────

    def _get_co2_emission(self) -> float:
        """Sum CO₂ emission (mg/s) of vehicles on controlled lanes."""
        total = 0.0
        for lid in self._ctrl_lanes:
            try:
                vids = traci.lane.getLastStepVehicleIDs(lid)
                for vid in vids:
                    total += traci.vehicle.getCO2Emission(vid)
            except Exception:
                pass
        return total

    def _vehicles_on_ctrl_lanes(
        self, vtype_set: set
    ) -> List[Tuple[str, str]]:
        """
        Return (vehicle_id, lane_id) pairs for vehicles of given types
        currently on any controlled lane or within approach distance.
        """
        result = []
        for lid in self._ctrl_lanes:
            try:
                vids = traci.lane.getLastStepVehicleIDs(lid)
            except Exception:
                continue
            for vid in vids:
                try:
                    if traci.vehicle.getTypeID(vid) in vtype_set:
                        result.append((vid, lid))
                except Exception:
                    pass
        return result

    def _dist_to_stop_line(self, vid: str, lid: str) -> float:
        """Approximate distance from a vehicle to the end of its lane."""
        try:
            lane_len = traci.lane.getLength(lid)
            veh_pos  = traci.vehicle.getLanePosition(vid)
            return max(0.0, lane_len - veh_pos)
        except Exception:
            return 9999.0

    def _bus_priority_bonus(self) -> float:
        """
        Bonus ∈ [0, 1]: how well the current phase serves approaching buses.
        = 1 if current phase is green for the lane the nearest bus is on.
        = 0 if no bus nearby or it is held at red.
        """
        buses = self._vehicles_on_ctrl_lanes(BUS_TYPES)
        if not buses:
            return 0.0

        # Find nearest bus
        nearest_bus, nearest_lane = min(
            buses,
            key=lambda x: self._dist_to_stop_line(x[0], x[1])
        )
        dist = self._dist_to_stop_line(nearest_bus, nearest_lane)
        if dist > self.rcfg.bus_dist_threshold:
            return 0.0

        # Check if current phase is green for that lane
        try:
            state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            lane_idx = self._ctrl_lanes.index(nearest_lane)
            if lane_idx < len(state) and state[lane_idx].lower() == "g":
                # Scale bonus by proximity (closer = higher bonus)
                proximity = 1.0 - dist / self.rcfg.bus_dist_threshold
                return float(proximity)
        except ValueError:
            pass
        return 0.0

    def _emergency_nearby(self) -> bool:
        """True if any emergency vehicle is within threshold distance."""
        evs = self._vehicles_on_ctrl_lanes(EMERGENCY_TYPES)
        for vid, lid in evs:
            if self._dist_to_stop_line(vid, lid) <= \
                    self.rcfg.emergency_dist_threshold:
                return True
        return False

    def _emergency_priority_bonus(self) -> float:
        """
        Bonus ∈ [0, 1]: reward if the current phase is green for the
        lane an emergency vehicle is approaching on.
        """
        evs = self._vehicles_on_ctrl_lanes(EMERGENCY_TYPES)
        if not evs:
            return 0.0

        nearest_ev, nearest_lane = min(
            evs,
            key=lambda x: self._dist_to_stop_line(x[0], x[1])
        )
        dist = self._dist_to_stop_line(nearest_ev, nearest_lane)
        if dist > self.rcfg.emergency_dist_threshold:
            return 0.0

        try:
            state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            lane_idx = self._ctrl_lanes.index(nearest_lane)
            if lane_idx < len(state) and state[lane_idx].lower() == "g":
                proximity = 1.0 - dist / self.rcfg.emergency_dist_threshold
                return float(proximity)
        except ValueError:
            pass
        return 0.0

    # ── Info dict ────────────────────────────────────────────────────────────

    def _info(self) -> dict:
        """Return a diagnostic info dict (not used for training)."""
        lane_results = traci.lane.getAllSubscriptionResults()
        return {
            "step":             self._step,
            "tls_id":           self.tls_id,
            "phase":            self._phase,
            "phase_timer":      self._phase_timer,
            "in_yellow":        self._in_yellow,
            "total_vehicles":   traci.vehicle.getIDCount(),
            "total_queue": sum(
                lane_results.get(lid, {}).get(
                    tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
                for lid in self._ctrl_lanes
            ),
            "total_waiting": sum(
                lane_results.get(lid, {}).get(tc.VAR_WAITING_TIME, 0.0)
                for lid in self._ctrl_lanes
            ),
            "emergency_nearby": self._emergency_nearby(),
        }


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== SmartCityTrafficEnv — Smoke Test ===")
    env = SmartCityTrafficEnv(
        use_gui   = False,
        max_steps = 100,
        port      = 8815,
    )

    obs, info = env.reset()
    print(f"Observation shape : {obs.shape}")
    print(f"Action space      : {env.action_space}")
    print(f"Obs space         : {env.observation_space}")
    print(f"TLS ID            : {env.tls_id}")
    print(f"Controlled lanes  : {len(env._ctrl_lanes)}")
    print(f"TLS phases        : {env._n_phases}")
    print()

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            print(
                f"  Step {step:3d} | action={action} | reward={reward:+.4f} "
                f"| queue={info['total_queue']:2d} "
                f"| wait={info['total_waiting']:.1f}s "
                f"| emerg={info['emergency_nearby']}"
            )

        if terminated or truncated:
            break

    print(f"\nTotal reward over episode: {total_reward:.4f}")
    env.close()
    print("=== Smoke test passed ===")
