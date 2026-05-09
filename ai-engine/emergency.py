"""
Emergency vehicle green-wave routing.

This module detects emergency vehicles in SUMO, finds the fastest route with
NetworkX A* search, and synchronizes traffic lights along the near-term route
as a green corridor. It is designed to run as a high-priority override before
TSP and PPO actions.
"""

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

log = logging.getLogger("emergency")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NET_PATH = SCRIPT_DIR.parent / "sumo" / "dhanbad.net.xml"

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for _p in ["/usr/share/sumo/tools", "/opt/sumo/tools"]:
        if os.path.isdir(_p):
            sys.path.append(_p)
            break


EMERGENCY_CLASSES = {"emergency", "authority", "vip"}
EMERGENCY_TYPES = {"emergency", "ambulance", "police", "fire", "rescue"}
INTERNAL_EDGE_PREFIX = ":"


@dataclass
class EmergencyVehicle:
    vehicle_id: str
    edge_id: str
    lane_id: str
    route: List[str]
    route_index: int
    speed: float
    waiting_time: float
    detected_at: float = field(default_factory=time.time)


@dataclass
class CorridorPlan:
    vehicle_id: str
    source_edge: str
    target_edge: str
    route_edges: List[str]
    total_cost: float
    eta_s: float


@dataclass
class CorridorActivation:
    active: bool
    vehicle_id: Optional[str] = None
    target_edge: Optional[str] = None
    route_edges: List[str] = field(default_factory=list)
    corridor_tls: List[str] = field(default_factory=list)
    applied_phases: Dict[str, int] = field(default_factory=dict)
    eta_s: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "active": self.active,
            "vehicle_id": self.vehicle_id,
            "target_edge": self.target_edge,
            "route_edges": self.route_edges,
            "corridor_tls": self.corridor_tls,
            "applied_phases": self.applied_phases,
            "eta_s": round(self.eta_s, 2),
            "reason": self.reason,
        }


class EmergencyVehicleSystem:
    """
    Builds and activates an emergency green wave.

    Workflow:
      1. Detect emergency vehicle position.
      2. Run NetworkX A* across SUMO edge connectivity.
      3. Select traffic lights on the route within the lookahead window.
      4. Force the phase that serves the route edge and hold it briefly.
    """

    def __init__(
        self,
        net_path: Optional[Path] = None,
        lookahead_seconds: float = 90.0,
        corridor_hold_seconds: float = 20.0,
        max_corridor_lights: int = 8,
        queue_penalty_s: float = 1.5,
        min_vehicle_speed: float = 5.0,
    ):
        self.net_path = Path(net_path or DEFAULT_NET_PATH)
        self.lookahead_seconds = lookahead_seconds
        self.corridor_hold_seconds = corridor_hold_seconds
        self.max_corridor_lights = max_corridor_lights
        self.queue_penalty_s = queue_penalty_s
        self.min_vehicle_speed = min_vehicle_speed

        self._nx = None
        self._graph = None
        self._edge_lookup = {}
        self._edge_positions: Dict[str, Tuple[float, float]] = {}
        self._route_cache: Dict[Tuple[str, str], CorridorPlan] = {}
        self._last_graph_error: Optional[str] = None
        self.activations = 0
        self.vehicles_helped: Set[str] = set()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_position(self, traci) -> Optional[EmergencyVehicle]:
        """Return the most urgent active emergency vehicle, if any."""
        candidates: List[EmergencyVehicle] = []
        for vehicle_id in traci.vehicle.getIDList():
            if not self._is_emergency_vehicle(traci, vehicle_id):
                continue
            try:
                edge_id = traci.vehicle.getRoadID(vehicle_id)
                lane_id = traci.vehicle.getLaneID(vehicle_id)
                if self._is_internal_edge(edge_id):
                    continue
                route = [
                    edge for edge in traci.vehicle.getRoute(vehicle_id)
                    if not self._is_internal_edge(edge)
                ]
                candidates.append(EmergencyVehicle(
                    vehicle_id=vehicle_id,
                    edge_id=edge_id,
                    lane_id=lane_id,
                    route=route,
                    route_index=int(traci.vehicle.getRouteIndex(vehicle_id)),
                    speed=float(traci.vehicle.getSpeed(vehicle_id)),
                    waiting_time=float(traci.vehicle.getWaitingTime(vehicle_id)),
                ))
            except Exception as exc:
                log.debug("Emergency detection skipped %s: %s", vehicle_id, exc)

        if not candidates:
            return None

        return max(candidates, key=lambda ev: (ev.waiting_time, -max(ev.speed, 0.0)))

    # ------------------------------------------------------------------
    # Route planning
    # ------------------------------------------------------------------
    def find_fastest_route(
        self,
        traci,
        emergency_vehicle: EmergencyVehicle,
        target_edge: Optional[str] = None,
    ) -> Optional[CorridorPlan]:
        """Find the fastest route from the vehicle's current edge to target."""
        source = emergency_vehicle.edge_id
        target = target_edge or self._destination_edge(emergency_vehicle)
        if not source or not target or source == target:
            return None

        if not self._ensure_graph():
            return None

        cache_key = (source, target)
        cached = self._route_cache.get(cache_key)
        if cached and cached.source_edge == source and cached.target_edge == target:
            return cached

        try:
            route_edges = self._nx.astar_path(
                self._graph,
                source,
                target,
                heuristic=lambda a, b: self._heuristic(a, b),
                weight=lambda u, v, d: self._live_weight(traci, u, v, d),
            )
            total_cost = sum(
                self._live_weight(traci, u, v, self._graph[u][v])
                for u, v in zip(route_edges, route_edges[1:])
            )
            eta_s = self._estimate_eta(traci, route_edges, emergency_vehicle.speed)
            plan = CorridorPlan(
                vehicle_id=emergency_vehicle.vehicle_id,
                source_edge=source,
                target_edge=target,
                route_edges=route_edges,
                total_cost=float(total_cost),
                eta_s=float(eta_s),
            )
            self._route_cache[cache_key] = plan
            return plan
        except Exception as exc:
            log.debug("A* route failed from %s to %s: %s", source, target, exc)
            return None

    # ------------------------------------------------------------------
    # Green corridor activation
    # ------------------------------------------------------------------
    def trigger_green_corridor(
        self,
        traci,
        vehicle_id: Optional[str] = None,
        target_edge: Optional[str] = None,
    ) -> CorridorActivation:
        """
        Detect or use a requested emergency vehicle and activate green wave.

        Returns a serializable activation summary for WebSocket/UI payloads.
        """
        emergency_vehicle = self._vehicle_from_id(traci, vehicle_id) if vehicle_id else None
        emergency_vehicle = emergency_vehicle or self.detect_position(traci)
        if not emergency_vehicle:
            return CorridorActivation(active=False, reason="no_emergency_vehicle")

        plan = self.find_fastest_route(traci, emergency_vehicle, target_edge)
        if not plan:
            return CorridorActivation(
                active=False,
                vehicle_id=emergency_vehicle.vehicle_id,
                reason=self._last_graph_error or "route_unavailable",
            )

        applied = self.synchronize_intersections(traci, plan, emergency_vehicle)
        active = bool(applied)
        if active:
            self.activations += 1
            self.vehicles_helped.add(emergency_vehicle.vehicle_id)

        return CorridorActivation(
            active=active,
            vehicle_id=emergency_vehicle.vehicle_id,
            target_edge=plan.target_edge,
            route_edges=plan.route_edges[:20],
            corridor_tls=list(applied.keys()),
            applied_phases=applied,
            eta_s=plan.eta_s,
            reason="green_corridor_active" if active else "no_signal_on_route",
        )

    def synchronize_intersections(
        self,
        traci,
        plan: CorridorPlan,
        emergency_vehicle: EmergencyVehicle,
    ) -> Dict[str, int]:
        """Set route-facing phases for corridor TLS within lookahead range."""
        route_edge_set = set(plan.route_edges)
        applied: Dict[str, int] = {}
        tls_candidates = self._traffic_lights_on_route(traci, plan.route_edges)

        for tls_id, route_index, incoming_edges in tls_candidates:
            if len(applied) >= self.max_corridor_lights:
                break
            eta = self._eta_to_route_index(
                traci,
                plan.route_edges,
                route_index,
                emergency_vehicle.speed,
            )
            if eta > self.lookahead_seconds:
                continue

            phase = self._best_green_phase_for_edges(traci, tls_id, route_edge_set, incoming_edges)
            if phase is None:
                continue

            try:
                traci.trafficlight.setPhase(tls_id, phase)
                try:
                    traci.trafficlight.setPhaseDuration(tls_id, self.corridor_hold_seconds)
                except Exception:
                    pass
                applied[tls_id] = int(phase)
            except Exception as exc:
                log.debug("Failed to set emergency corridor phase for %s: %s", tls_id, exc)

        return applied

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_graph(self) -> bool:
        if self._graph is not None:
            return True

        try:
            import networkx as nx
        except ImportError:
            self._last_graph_error = "networkx_missing"
            log.warning("NetworkX is required for emergency A* routing")
            return False

        try:
            # pyrefly: ignore [missing-import]
            from sumolib.net import readNet
        except ImportError:
            self._last_graph_error = "sumolib_missing"
            log.warning("sumolib is required to read %s", self.net_path)
            return False

        if not self.net_path.exists():
            self._last_graph_error = f"net_missing:{self.net_path}"
            log.warning("SUMO net file not found: %s", self.net_path)
            return False

        try:
            net = readNet(str(self.net_path))
            graph = nx.DiGraph()
            for edge in net.getEdges():
                edge_id = edge.getID()
                if self._is_internal_edge(edge_id):
                    continue

                self._edge_lookup[edge_id] = edge
                self._edge_positions[edge_id] = self._edge_position(edge)
                graph.add_node(edge_id)

            for edge_id, edge in self._edge_lookup.items():
                outgoing = edge.getOutgoing()
                outgoing_edges = outgoing.keys() if hasattr(outgoing, "keys") else outgoing
                for out_edge in outgoing_edges:
                    out_id = out_edge.getID()
                    if self._is_internal_edge(out_id) or out_id not in self._edge_lookup:
                        continue
                    graph.add_edge(
                        edge_id,
                        out_id,
                        base_weight=self._base_travel_time(out_edge),
                        length=max(float(out_edge.getLength()), 1.0),
                    )

            self._nx = nx
            self._graph = graph
            self._last_graph_error = None
            log.info("Emergency graph ready: %s edges, %s links", graph.number_of_nodes(), graph.number_of_edges())
            return True
        except Exception as exc:
            self._last_graph_error = f"graph_build_failed:{exc}"
            log.warning("Failed to build emergency route graph: %s", exc)
            return False

    def _is_emergency_vehicle(self, traci, vehicle_id: str) -> bool:
        lowered = vehicle_id.lower()
        if any(token in lowered for token in EMERGENCY_TYPES):
            return True
        try:
            vclass = traci.vehicle.getVehicleClass(vehicle_id).lower()
            if vclass in EMERGENCY_CLASSES or vclass in EMERGENCY_TYPES:
                return True
        except Exception:
            pass
        try:
            type_id = traci.vehicle.getTypeID(vehicle_id).lower()
            return any(token in type_id for token in EMERGENCY_TYPES)
        except Exception:
            return False

    def _vehicle_from_id(self, traci, vehicle_id: str) -> Optional[EmergencyVehicle]:
        if vehicle_id not in traci.vehicle.getIDList():
            return None
        if not self._is_emergency_vehicle(traci, vehicle_id):
            return None
        try:
            route = [
                edge for edge in traci.vehicle.getRoute(vehicle_id)
                if not self._is_internal_edge(edge)
            ]
            return EmergencyVehicle(
                vehicle_id=vehicle_id,
                edge_id=traci.vehicle.getRoadID(vehicle_id),
                lane_id=traci.vehicle.getLaneID(vehicle_id),
                route=route,
                route_index=int(traci.vehicle.getRouteIndex(vehicle_id)),
                speed=float(traci.vehicle.getSpeed(vehicle_id)),
                waiting_time=float(traci.vehicle.getWaitingTime(vehicle_id)),
            )
        except Exception:
            return None

    def _destination_edge(self, emergency_vehicle: EmergencyVehicle) -> Optional[str]:
        if not emergency_vehicle.route:
            return None
        for edge in reversed(emergency_vehicle.route):
            if not self._is_internal_edge(edge):
                return edge
        return None

    def _traffic_lights_on_route(
        self,
        traci,
        route_edges: Sequence[str],
    ) -> List[Tuple[str, int, Set[str]]]:
        route_index = {edge: idx for idx, edge in enumerate(route_edges)}
        candidates = []

        for tls_id in traci.trafficlight.getIDList():
            incoming_edges: Set[str] = set()
            try:
                ctrl_links = traci.trafficlight.getControlledLinks(tls_id)
                for group in ctrl_links:
                    for link in group:
                        if not link:
                            continue
                        incoming_lane = link[0]
                        incoming_edge = self._lane_edge_id(traci, incoming_lane)
                        if incoming_edge in route_index:
                            incoming_edges.add(incoming_edge)
            except Exception as exc:
                log.debug("TLS route scan skipped %s: %s", tls_id, exc)

            if incoming_edges:
                candidates.append((tls_id, min(route_index[e] for e in incoming_edges), incoming_edges))

        return sorted(candidates, key=lambda item: item[1])

    def _best_green_phase_for_edges(
        self,
        traci,
        tls_id: str,
        route_edge_set: Set[str],
        incoming_edges: Set[str],
    ) -> Optional[int]:
        try:
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            ctrl_links = traci.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return None

        phase_scores: List[Tuple[int, int]] = []
        for phase_idx, phase in enumerate(logic.phases):
            state = phase.state
            score = 0
            for signal_idx, group in enumerate(ctrl_links):
                if signal_idx >= len(state) or state[signal_idx] not in ("g", "G"):
                    continue
                for link in group:
                    if not link:
                        continue
                    incoming_edge = self._lane_edge_id(traci, link[0])
                    outgoing_edge = self._lane_edge_id(traci, link[1]) if len(link) > 1 else ""
                    if incoming_edge in incoming_edges:
                        score += 3
                    elif incoming_edge in route_edge_set or outgoing_edge in route_edge_set:
                        score += 1
            if score > 0:
                phase_scores.append((score, phase_idx))

        if not phase_scores:
            return None
        return max(phase_scores, key=lambda item: item[0])[1]

    def _live_weight(self, traci, _u: str, v: str, data: Dict) -> float:
        base = float(data.get("base_weight", 1.0))
        try:
            travel_time = float(traci.edge.getTraveltime(v))
            if math.isfinite(travel_time) and travel_time > 0:
                base = travel_time
        except Exception:
            pass
        try:
            queue = float(traci.edge.getLastStepHaltingNumber(v))
            base += queue * self.queue_penalty_s
        except Exception:
            pass
        return max(base, 0.1)

    def _estimate_eta(self, traci, route_edges: Sequence[str], fallback_speed: float) -> float:
        if len(route_edges) <= 1:
            return 0.0
        return self._eta_to_route_index(traci, route_edges, len(route_edges) - 1, fallback_speed)

    def _eta_to_route_index(
        self,
        traci,
        route_edges: Sequence[str],
        target_index: int,
        fallback_speed: float,
    ) -> float:
        speed = max(float(fallback_speed or 0.0), self.min_vehicle_speed)
        eta = 0.0
        for edge_id in route_edges[:max(target_index, 0)]:
            length = self._edge_length(traci, edge_id)
            eta += length / speed
        return eta

    def _heuristic(self, edge_a: str, edge_b: str) -> float:
        ax, ay = self._edge_positions.get(edge_a, (0.0, 0.0))
        bx, by = self._edge_positions.get(edge_b, (0.0, 0.0))
        return math.hypot(ax - bx, ay - by) / 16.67

    def _base_travel_time(self, edge) -> float:
        length = max(float(edge.getLength()), 1.0)
        speed = max(float(edge.getSpeed() or 13.9), 1.0)
        return length / speed

    def _edge_position(self, edge) -> Tuple[float, float]:
        try:
            return tuple(float(v) for v in edge.getToNode().getCoord())
        except Exception:
            try:
                shape = edge.getShape()
                if shape:
                    return tuple(float(v) for v in shape[-1])
            except Exception:
                pass
        return (0.0, 0.0)

    def _edge_length(self, traci, edge_id: str) -> float:
        edge = self._edge_lookup.get(edge_id)
        if edge is not None:
            try:
                return max(float(edge.getLength()), 1.0)
            except Exception:
                pass
        try:
            lane_count = traci.edge.getLaneNumber(edge_id)
            if lane_count > 0:
                return max(float(traci.lane.getLength(f"{edge_id}_0")), 1.0)
        except Exception:
            pass
        return 100.0

    def _lane_edge_id(self, traci, lane_id: str) -> str:
        try:
            return traci.lane.getEdgeID(lane_id)
        except Exception:
            return lane_id.rsplit("_", 1)[0]

    def _is_internal_edge(self, edge_id: Optional[str]) -> bool:
        return not edge_id or edge_id.startswith(INTERNAL_EDGE_PREFIX)

    def get_stats(self) -> Dict:
        return {
            "activations": self.activations,
            "vehicles_helped": len(self.vehicles_helped),
            "graph_ready": self._graph is not None,
            "last_graph_error": self._last_graph_error,
        }


emergency_system = EmergencyVehicleSystem()
