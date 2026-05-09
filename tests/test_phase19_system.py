import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = PROJECT_DIR / "ai-engine"
sys.path.insert(0, str(AI_DIR))

from bus_priority import BusPrioritySystem
from emissions import CopertIVEmissionModel
from emergency import EmergencyVehicleSystem
from multi_agent import MultiIntersectionCoordinator


class FakePhase:
    def __init__(self, state):
        self.state = state


class FakeLogic:
    def __init__(self, states):
        self.phases = [FakePhase(state) for state in states]


class FakeTrafficLight:
    def __init__(self):
        self.links = {
            "tlsA": [
                [("laneA_0", "laneB_0", None)],
                [("laneX_0", "laneY_0", None)],
            ],
            "tlsB": [
                [("laneB_0", "laneC_0", None)],
                [("laneY_0", "laneZ_0", None)],
            ],
        }
        self.phases = {
            "tlsA": FakeLogic(["Gr", "rG"]),
            "tlsB": FakeLogic(["Gr", "rG"]),
        }
        self.current_phase = {"tlsA": 0, "tlsB": 0}
        self.applied = {}
        self.phase_durations = {}

    def getIDList(self):
        return list(self.links.keys())

    def getControlledLinks(self, tls_id):
        return self.links[tls_id]

    def getAllProgramLogics(self, tls_id):
        return [self.phases[tls_id]]

    def getPhase(self, tls_id):
        return self.current_phase[tls_id]

    def setPhase(self, tls_id, phase):
        self.current_phase[tls_id] = phase
        self.applied[tls_id] = phase

    def setPhaseDuration(self, tls_id, duration):
        self.phase_durations[tls_id] = duration

    def getRedYellowGreenState(self, tls_id):
        return self.phases[tls_id].phases[self.current_phase[tls_id]].state


class FakeLane:
    def __init__(self):
        self.vehicles = {
            "laneA_0": ["bus_1"],
            "laneX_0": [],
            "laneB_0": [],
            "laneY_0": [],
        }
        self.lengths = {lane: 200.0 for lane in self.vehicles}

    def getLastStepVehicleIDs(self, lane_id):
        return self.vehicles.get(lane_id, [])

    def getLength(self, lane_id):
        return self.lengths.get(lane_id, 200.0)

    def getEdgeID(self, lane_id):
        return lane_id.rsplit("_", 1)[0]


class FakeVehicle:
    def getVehicleClass(self, vehicle_id):
        if "bus" in vehicle_id:
            return "bus"
        return "passenger"

    def getLanePosition(self, vehicle_id):
        return 80.0

    def getWaitingTime(self, vehicle_id):
        return 20.0


class FakeTraci:
    def __init__(self):
        self.trafficlight = FakeTrafficLight()
        self.lane = FakeLane()
        self.vehicle = FakeVehicle()


class FakeModel:
    def predict(self, obs, deterministic=True):
        queue_hint = obs[0] if len(obs) else 0
        return (1 if queue_hint > 0.2 else 0), None


def build_obs(telemetry, tls_id):
    queue = telemetry["intersections"][tls_id]["total_queue"]
    return [min(queue / 20.0, 1.0)]


class Phase19SystemTests(unittest.TestCase):
    def test_rl_agent_outputs_valid_actions_for_multiple_junctions(self):
        traci = FakeTraci()
        telemetry = {
            "intersections": {
                "tlsA": {"total_queue": 10, "total_waiting": 80, "phase_index": 0},
                "tlsB": {"total_queue": 2, "total_waiting": 10, "phase_index": 0},
            }
        }

        coordinator = MultiIntersectionCoordinator(
            min_switch_interval_s=0,
            green_wave_queue_threshold=99,
        )
        result = coordinator.control_step(
            traci=traci,
            telemetry=telemetry,
            tls_ids=["tlsA", "tlsB"],
            model=FakeModel(),
            obs_builder=build_obs,
        )

        self.assertTrue(result.actions)
        self.assertTrue(all(action in (0, 1, 2, 3) for action in result.actions.values()))
        self.assertIn("tlsA", result.actions)
        self.assertIn("tlsB", result.actions)

    def test_multi_junction_sync_prevents_neighbor_switch_conflict(self):
        traci = FakeTraci()
        traci.lane.vehicles["laneA_0"] = []
        telemetry = {
            "intersections": {
                "tlsA": {"total_queue": 10, "total_waiting": 80, "phase_index": 0},
                "tlsB": {"total_queue": 9, "total_waiting": 70, "phase_index": 0},
            }
        }

        coordinator = MultiIntersectionCoordinator(
            min_switch_interval_s=0,
            max_switches_per_step=4,
            green_wave_queue_threshold=99,
        )
        result = coordinator.control_step(
            traci=traci,
            telemetry=telemetry,
            tls_ids=["tlsA", "tlsB"],
            model=FakeModel(),
            obs_builder=build_obs,
        )

        switched = [tls for tls, action in result.actions.items() if action in (1, 2)]
        self.assertEqual(len(switched), 1)
        self.assertIn("neighbor_switch_conflict", result.skipped.values())

    def test_bus_priority_override_switches_delayed_bus_on_red(self):
        traci = FakeTraci()
        traci.trafficlight.current_phase["tlsA"] = 1

        action = BusPrioritySystem(max_distance=200, delay_threshold=10).get_priority_action(
            traci,
            "tlsA",
        )

        self.assertEqual(action, 1)

    def test_emergency_priority_selects_route_facing_green_phase(self):
        traci = FakeTraci()
        system = EmergencyVehicleSystem()

        phase = system._best_green_phase_for_edges(
            traci=traci,
            tls_id="tlsA",
            route_edge_set={"laneA", "laneB"},
            incoming_edges={"laneA"},
        )

        self.assertEqual(phase, 0)

    def test_green_wave_optimization_marks_congested_corridor(self):
        coordinator = MultiIntersectionCoordinator(green_wave_queue_threshold=5)
        coordinator._neighbors = {"tlsA": {"tlsB"}, "tlsB": {"tlsA"}}

        decisions = coordinator._apply_green_wave(
            decisions=[
                type("Decision", (), {
                    "tls_id": "tlsA",
                    "action": 1,
                    "source": "ppo",
                    "queue": 12,
                    "waiting": 90,
                    "phase_index": 0,
                })(),
                type("Decision", (), {
                    "tls_id": "tlsB",
                    "action": 1,
                    "source": "ppo",
                    "queue": 8,
                    "waiting": 50,
                    "phase_index": 0,
                })(),
            ],
            telemetry={
                "intersections": {
                    "tlsA": {"total_queue": 12, "total_waiting": 90},
                    "tlsB": {"total_queue": 8, "total_waiting": 50},
                }
            },
        )

        sources = {decision.tls_id: decision.source for decision in decisions}
        self.assertEqual(sources["tlsA"], "green_wave")
        self.assertEqual(sources["tlsB"], "green_wave")

    def test_copert_emission_model_reports_reduction_against_baseline(self):
        model = CopertIVEmissionModel()
        lane = model.estimate_lane(
            speed_mps=8.0,
            vehicle_count=6,
            queue_count=2,
            step_length_s=1.0,
            class_counts={"passenger": 5, "bus": 1},
        )

        self.assertGreater(lane["co2_rate_g_s"], 0)
        self.assertGreater(model.reduction_pct(lane["co2_rate_g_s"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
