"""
COPERT IV-style CO2 emission analytics.

The hot-emission estimate follows the common COPERT shape:

    CO2 = EF(v) * distance

where EF(v) is a speed-dependent emission factor in g/km and distance is the
distance travelled during the simulation step in km. Queued vehicles are given
an idle-emission add-on so stop-and-go pollution hotspots show up quickly.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class CopertCoefficients:
    a: float
    b: float
    c: float
    d: float


class CopertIVEmissionModel:
    """
    Lightweight COPERT IV-style model for live SUMO telemetry.

    Coefficients approximate speed-sensitive CO2 factors for urban traffic.
    They are intentionally conservative and can be replaced by calibrated local
    fleet factors later without touching the backend/frontend contract.
    """

    DEFAULT_CLASS = "passenger"

    COEFFICIENTS: Dict[str, CopertCoefficients] = {
        "passenger": CopertCoefficients(a=146.0, b=-1.08, c=0.018, d=750.0),
        "bus":       CopertCoefficients(a=780.0, b=-4.20, c=0.045, d=5200.0),
        "coach":     CopertCoefficients(a=780.0, b=-4.20, c=0.045, d=5200.0),
        "truck":     CopertCoefficients(a=620.0, b=-3.10, c=0.040, d=4100.0),
        "delivery":  CopertCoefficients(a=240.0, b=-1.50, c=0.022, d=1200.0),
        "taxi":      CopertCoefficients(a=156.0, b=-1.10, c=0.019, d=820.0),
        "motorcycle":CopertCoefficients(a=70.0,  b=-0.35, c=0.006, d=300.0),
        "emergency": CopertCoefficients(a=260.0, b=-1.60, c=0.024, d=1600.0),
    }

    IDLE_G_PER_S: Dict[str, float] = {
        "passenger": 0.62,
        "bus": 4.40,
        "coach": 4.40,
        "truck": 3.80,
        "delivery": 1.10,
        "taxi": 0.72,
        "motorcycle": 0.24,
        "emergency": 1.35,
    }

    BASELINE_CO2_G_PER_S = 624.82

    def emission_factor(self, speed_mps: float, vehicle_class: str = DEFAULT_CLASS) -> float:
        """Return COPERT-style hot CO2 emission factor in g/km."""
        speed_kmh = max(float(speed_mps) * 3.6, 1.0)
        coeffs = self.COEFFICIENTS.get(
            self._normalize_class(vehicle_class),
            self.COEFFICIENTS[self.DEFAULT_CLASS],
        )
        ef = coeffs.a + coeffs.b * speed_kmh + coeffs.c * speed_kmh ** 2 + coeffs.d / speed_kmh
        return max(ef, 45.0)

    def estimate_lane(
        self,
        speed_mps: float,
        vehicle_count: int,
        queue_count: int,
        step_length_s: float = 1.0,
        class_counts: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, float]:
        """Estimate lane CO2 for one simulation step."""
        vehicle_count = max(int(vehicle_count or 0), 0)
        queue_count = max(int(queue_count or 0), 0)
        speed_mps = max(float(speed_mps or 0.0), 0.0)
        step_length_s = max(float(step_length_s or 1.0), 0.1)

        classes = self._expanded_class_counts(class_counts, vehicle_count)
        distance_km = speed_mps * step_length_s / 1000.0

        moving_co2_g = 0.0
        idle_weighted_g_s = 0.0
        for vehicle_class, count in classes.items():
            normalized = self._normalize_class(vehicle_class)
            moving_co2_g += self.emission_factor(speed_mps, normalized) * distance_km * count
            idle_weighted_g_s += self.IDLE_G_PER_S.get(
                normalized,
                self.IDLE_G_PER_S[self.DEFAULT_CLASS],
            ) * count

        queued_share = min(queue_count / vehicle_count, 1.0) if vehicle_count else 0.0
        idle_co2_g = idle_weighted_g_s * queued_share * step_length_s
        total_co2_g = moving_co2_g + idle_co2_g

        return {
            "co2_g": round(total_co2_g, 3),
            "co2_rate_g_s": round(total_co2_g / step_length_s, 3),
            "co2_per_vehicle_g": round(total_co2_g / vehicle_count, 3) if vehicle_count else 0.0,
            "emission_intensity": round(min(total_co2_g / 25.0, 1.0), 3),
        }

    def reduction_pct(self, co2_rate_g_s: float, baseline_g_s: float = BASELINE_CO2_G_PER_S) -> float:
        """Positive percentage means live AI is emitting less than baseline."""
        if baseline_g_s <= 0:
            return 0.0
        return round((baseline_g_s - co2_rate_g_s) / baseline_g_s * 100.0, 1)

    def _expanded_class_counts(
        self,
        class_counts: Optional[Mapping[str, int]],
        vehicle_count: int,
    ) -> Dict[str, int]:
        if not class_counts:
            return {self.DEFAULT_CLASS: vehicle_count}

        expanded: Dict[str, int] = {}
        counted = 0
        for vehicle_class, count in class_counts.items():
            safe_count = max(int(count or 0), 0)
            if safe_count:
                expanded[self._normalize_class(vehicle_class)] = expanded.get(
                    self._normalize_class(vehicle_class), 0
                ) + safe_count
                counted += safe_count

        if vehicle_count > counted:
            expanded[self.DEFAULT_CLASS] = expanded.get(self.DEFAULT_CLASS, 0) + (vehicle_count - counted)
        return expanded or {self.DEFAULT_CLASS: vehicle_count}

    def _normalize_class(self, vehicle_class: str) -> str:
        value = (vehicle_class or self.DEFAULT_CLASS).lower()
        if value in ("private", "passenger", "car", "hov"):
            return "passenger"
        if value in ("bus", "coach"):
            return value
        if value in ("truck", "trailer", "heavy", "hov_truck"):
            return "truck"
        if value in ("delivery", "van"):
            return "delivery"
        if value in ("taxi",):
            return "taxi"
        if value in ("motorcycle", "moped"):
            return "motorcycle"
        if value in ("emergency", "authority", "police", "fire", "ambulance"):
            return "emergency"
        return self.DEFAULT_CLASS


emission_model = CopertIVEmissionModel()
