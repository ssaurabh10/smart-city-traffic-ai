"""
Multi-intersection coordination for smart-city traffic control.

Each traffic light is treated as an agent that can request a local PPO action.
The coordinator then applies network-level rules:

  * bus-priority overrides remain local but can happen at any junction
  * adjacent junctions are throttled so they do not all switch at once
  * a queue-driven green wave is formed across the most congested corridor
  * phase changes are rate-limited for deployment stability
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

log = logging.getLogger("multi-agent")


@dataclass
class AgentDecision:
    tls_id: str
    action: int
    source: str
    queue: int = 0
    waiting: float = 0.0
    phase_index: int = -1
    reason: str = ""


@dataclass
class CoordinationResult:
    actions: Dict[str, int] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    coordinated_tls: List[str] = field(default_factory=list)
    green_wave: Dict = field(default_factory=dict)
    bus_priority_active: bool = False
    skipped: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        payload = dict(self.actions)
        payload.update({
            "multi_agent_active": True,
            "coordination_active": bool(self.coordinated_tls),
            "coordinated_tls": self.coordinated_tls,
            "action_sources": self.sources,
            "green_wave": self.green_wave,
            "green_wave_active": bool(self.green_wave.get("active")),
            "bus_priority_active": self.bus_priority_active,
            "skipped": self.skipped,
        })
        return payload


class MultiIntersectionCoordinator:
    def __init__(
        self,
        min_switch_interval_s: float = 8.0,
        max_switches_per_step: int = 4,
        green_wave_horizon: int = 5,
        green_wave_queue_threshold: int = 6,
        green_wave_hold_s: float = 18.0,
    ):
        self.min_switch_interval_s = min_switch_interval_s
        self.max_switches_per_step = max_switches_per_step
        self.green_wave_horizon = green_wave_horizon
        self.green_wave_queue_threshold = green_wave_queue_threshold
        self.green_wave_hold_s = green_wave_hold_s
        self._last_switch_at: Dict[str, float] = {}
        self._neighbors: Dict[str, Set[str]] = {}
        self._tls_order: List[str] = []

    def control_step(
        self,
        traci,
        telemetry: Dict,
        tls_ids: Iterable[str],
        model,
        obs_builder: Callable[[Dict, str], object],
    ) -> CoordinationResult:
        """Predict and coordinate one action for every intersection agent."""
        tls_ids = list(tls_ids)
        if not tls_ids:
            return CoordinationResult()

        self._refresh_topology(traci, tls_ids)

        decisions = self._local_agent_decisions(traci, telemetry, tls_ids, model, obs_builder)
        decisions = self._apply_green_wave(decisions, telemetry)
        return self._apply_coordinated_actions(traci, decisions)

    # ------------------------------------------------------------------
    # Local agent layer
    # ------------------------------------------------------------------
    def _local_agent_decisions(
        self,
        traci,
        telemetry: Dict,
        tls_ids: List[str],
        model,
        obs_builder: Callable[[Dict, str], object],
    ) -> List[AgentDecision]:
        from bus_priority import tsp_system

        decisions: List[AgentDecision] = []
        intersections = telemetry.get("intersections", {})

        for tls_id in tls_ids:
            im = intersections.get(tls_id, {})
            action: Optional[int] = None
            source = "ppo"
            reason = "local_policy"

            try:
                action = tsp_system.get_priority_action(traci, tls_id)
            except Exception as exc:
                log.debug("TSP skipped for %s: %s", tls_id, exc)

            if action is not None:
                source = "bus_priority"
                reason = "transit_priority"
            else:
                try:
                    agent_action, _ = model.predict(obs_builder(telemetry, tls_id), deterministic=True)
                    action = int(agent_action)
                except Exception as exc:
                    log.debug("PPO prediction failed for %s: %s", tls_id, exc)
                    action = 0
                    source = "fallback"
                    reason = "prediction_failed"

            decisions.append(AgentDecision(
                tls_id=tls_id,
                action=int(action),
                source=source,
                queue=int(im.get("total_queue", 0)),
                waiting=float(im.get("total_waiting", 0.0)),
                phase_index=int(im.get("phase_index", -1)),
                reason=reason,
            ))

        return decisions

    # ------------------------------------------------------------------
    # Coordination layer
    # ------------------------------------------------------------------
    def _apply_green_wave(
        self,
        decisions: List[AgentDecision],
        telemetry: Dict,
    ) -> List[AgentDecision]:
        if not decisions:
            return decisions

        intersections = telemetry.get("intersections", {})
        root = max(
            decisions,
            key=lambda d: (d.queue, d.waiting),
        )
        if root.queue < self.green_wave_queue_threshold:
            return decisions

        corridor = self._corridor_from(root.tls_id, intersections)
        if len(corridor) < 2:
            return decisions

        corridor_set = set(corridor)
        updated: List[AgentDecision] = []
        for decision in decisions:
            if decision.tls_id in corridor_set and decision.source == "ppo":
                updated.append(AgentDecision(
                    tls_id=decision.tls_id,
                    action=0,
                    source="green_wave",
                    queue=decision.queue,
                    waiting=decision.waiting,
                    phase_index=decision.phase_index,
                    reason=f"coordinated_corridor:{root.tls_id}",
                ))
            else:
                updated.append(decision)
        return updated

    def _apply_coordinated_actions(self, traci, decisions: List[AgentDecision]) -> CoordinationResult:
        result = CoordinationResult()
        now = time.time()
        switch_budget = self.max_switches_per_step

        prioritized = sorted(
            decisions,
            key=lambda d: (
                0 if d.source in ("bus_priority", "green_wave") else 1,
                -d.queue,
                -d.waiting,
            ),
        )

        for decision in prioritized:
            action = int(decision.action)
            tls_id = decision.tls_id

            if action in (1, 2):
                if switch_budget <= 0:
                    result.skipped[tls_id] = "switch_budget_exhausted"
                    continue
                if not self._can_switch(tls_id, now):
                    result.skipped[tls_id] = "switch_cooldown"
                    action = 0
                elif self._neighbor_switched(tls_id, result.actions):
                    result.skipped[tls_id] = "neighbor_switch_conflict"
                    action = 0

            applied = self._apply_action(traci, tls_id, action, decision.source)
            if not applied:
                result.skipped[tls_id] = "apply_failed"
                continue

            result.actions[tls_id] = action
            result.sources[tls_id] = decision.source
            if decision.source in ("green_wave", "bus_priority"):
                result.coordinated_tls.append(tls_id)
            if decision.source == "bus_priority":
                result.bus_priority_active = True
            if action in (1, 2):
                self._last_switch_at[tls_id] = now
                switch_budget -= 1

        green_wave_tls = [
            d.tls_id for d in decisions
            if d.source == "green_wave" and d.tls_id in result.actions
        ]
        if green_wave_tls:
            result.green_wave = {
                "active": True,
                "tls_ids": green_wave_tls,
                "hold_s": self.green_wave_hold_s,
            }
        else:
            result.green_wave = {"active": False, "tls_ids": []}

        return result

    def _apply_action(self, traci, tls_id: str, action: int, source: str) -> bool:
        try:
            if action in (1, 2):
                logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                cur = traci.trafficlight.getPhase(tls_id)
                n = len(logic.phases)
                if n > 1:
                    traci.trafficlight.setPhase(tls_id, (cur + action) % n)
            elif action == 3:
                state_len = len(traci.trafficlight.getRedYellowGreenState(tls_id))
                traci.trafficlight.setRedYellowGreenState(tls_id, "r" * state_len)
            elif source == "green_wave":
                try:
                    traci.trafficlight.setPhaseDuration(tls_id, self.green_wave_hold_s)
                except Exception:
                    pass
            return True
        except Exception as exc:
            log.debug("Action apply failed for %s: %s", tls_id, exc)
            return False

    # ------------------------------------------------------------------
    # Topology / green-wave heuristics
    # ------------------------------------------------------------------
    def _refresh_topology(self, traci, tls_ids: List[str]) -> None:
        if self._tls_order == tls_ids and self._neighbors:
            return

        self._tls_order = list(tls_ids)
        lane_to_tls: Dict[str, str] = {}
        for tls_id in tls_ids:
            try:
                for group in traci.trafficlight.getControlledLinks(tls_id):
                    for link in group:
                        if link:
                            lane_to_tls[link[0]] = tls_id
            except Exception:
                continue

        neighbors: Dict[str, Set[str]] = {tls_id: set() for tls_id in tls_ids}
        for tls_id in tls_ids:
            try:
                for group in traci.trafficlight.getControlledLinks(tls_id):
                    for link in group:
                        if not link or len(link) < 2:
                            continue
                        next_tls = lane_to_tls.get(link[1])
                        if next_tls and next_tls != tls_id:
                            neighbors[tls_id].add(next_tls)
                            neighbors[next_tls].add(tls_id)
            except Exception:
                continue

        # Sparse SUMO networks can hide adjacency behind intermediate lanes.
        # Keep a stable fallback chain so green-wave coordination still scales.
        for left, right in zip(tls_ids, tls_ids[1:]):
            neighbors[left].add(right)
            neighbors[right].add(left)

        self._neighbors = neighbors

    def _corridor_from(self, root_tls: str, intersections: Dict) -> List[str]:
        corridor = [root_tls]
        visited = {root_tls}
        current = root_tls

        while len(corridor) < self.green_wave_horizon:
            candidates = [
                tls_id for tls_id in self._neighbors.get(current, set())
                if tls_id not in visited
            ]
            if not candidates:
                break
            current = max(
                candidates,
                key=lambda tls_id: (
                    intersections.get(tls_id, {}).get("total_queue", 0),
                    intersections.get(tls_id, {}).get("total_waiting", 0.0),
                ),
            )
            corridor.append(current)
            visited.add(current)

        return corridor

    def _can_switch(self, tls_id: str, now: float) -> bool:
        return now - self._last_switch_at.get(tls_id, 0.0) >= self.min_switch_interval_s

    def _neighbor_switched(self, tls_id: str, applied_actions: Dict[str, int]) -> bool:
        return any(
            applied_actions.get(neighbor) in (1, 2)
            for neighbor in self._neighbors.get(tls_id, set())
        )


multi_agent_coordinator = MultiIntersectionCoordinator()
