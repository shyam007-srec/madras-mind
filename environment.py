from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import libsumo as sumo_backend  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sumo_backend = None

try:
    import traci  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    traci = None


@dataclass
class CorridorCalibration:
    route_geometry_path: Optional[str] = None
    traffic_count_path: Optional[str] = None
    demand_scale: float = 1.0
    vehicle_split: Dict[str, float] = field(
        default_factory=lambda: {
            "two_wheeler": 0.60,
            "car": 0.25,
            "auto": 0.10,
            "bus_heavy": 0.05,
        }
    )
    min_green_seconds: int = 8
    yellow_seconds: int = 3
    red_seconds: int = 1
    occupancy_m2_per_vehicle: Dict[str, float] = field(
        default_factory=lambda: {
            "two_wheeler": 1.8,
            "car": 7.5,
            "auto": 5.5,
            "bus_heavy": 18.0,
        }
    )
    phase_plan: Tuple[Tuple[str, int], ...] = (
        ("phase_0", 30),
        ("phase_1", 28),
        ("phase_2", 24),
        ("phase_3", 32),
    )


@dataclass
class IntersectionState:
    intersection_id: str
    queue_length: float = 0.0
    incoming_capacity: float = 1.0
    outgoing_capacity: float = 1.0
    occupancy_m2: float = 0.0
    current_phase: int = 0
    phase_age: int = 0
    in_yellow: bool = False
    sensor_blind: bool = False
    pressure: float = 0.0
    neighbor_inference: Dict[str, float] = field(default_factory=dict)
    demand_profile: Dict[str, float] = field(default_factory=dict)
    phase_history: Deque[int] = field(default_factory=lambda: deque(maxlen=8))


@dataclass
class DecisionLogEntry:
    timestep: int
    intersection_id: str
    action: int
    pressure: float
    inferred_neighbor_state: Dict[str, float]
    state_vector: List[float]


class DecisionAuditLogger:
    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.records: List[DecisionLogEntry] = []

    def log(self, entry: DecisionLogEntry) -> None:
        self.records.append(entry)
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.__dict__) + "\n")


class MockSumoBackend:
    def __init__(
        self,
        intersections: Sequence[str],
        calibration: CorridorCalibration,
        seed: int = 7,
    ) -> None:
        self.calibration = calibration
        self.random = random.Random(seed)
        self.intersection_ids = list(intersections)
        self.states = {
            intersection_id: IntersectionState(
                intersection_id=intersection_id,
                incoming_capacity=12.0,
                outgoing_capacity=11.0,
                demand_profile=self._build_demand_profile(intersection_id),
            )
            for intersection_id in self.intersection_ids
        }
        self.neighbors = self._build_neighbor_graph(self.intersection_ids)
        self.timestep = 0

    def _build_neighbor_graph(self, intersections: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
        neighbor_graph: Dict[str, Tuple[str, ...]] = {}
        for index, intersection_id in enumerate(intersections):
            left = intersections[index - 1] if index > 0 else intersections[index]
            right = intersections[index + 1] if index < len(intersections) - 1 else intersections[index]
            neighbor_graph[intersection_id] = tuple(dict.fromkeys((left, right)))
        return neighbor_graph

    def _build_demand_profile(self, intersection_id: str) -> Dict[str, float]:
        base_count = 90.0 * self.calibration.demand_scale
        profile = {
            vehicle_class: base_count * share
            for vehicle_class, share in self.calibration.vehicle_split.items()
        }
        profile["intersection_id"] = intersection_id
        return profile

    def reset(self) -> Dict[str, IntersectionState]:
        self.timestep = 0
        for state in self.states.values():
            state.queue_length = self.random.uniform(5.0, 25.0)
            state.occupancy_m2 = state.queue_length * 7.0
            state.current_phase = 0
            state.phase_age = 0
            state.in_yellow = False
            state.sensor_blind = False
            state.pressure = 0.0
            state.neighbor_inference = {}
            state.phase_history.clear()
        return self.states

    def step(self, actions: Dict[str, int]) -> Dict[str, IntersectionState]:
        self.timestep += 1
        for intersection_id, state in self.states.items():
            proposed_action = int(actions.get(intersection_id, state.current_phase))
            if proposed_action != state.current_phase and state.phase_age < self.calibration.min_green_seconds:
                proposed_action = state.current_phase
                state.in_yellow = False
            elif proposed_action != state.current_phase:
                state.in_yellow = True
                state.phase_age = 0
                state.current_phase = proposed_action

            if state.in_yellow:
                if state.phase_age >= self.calibration.yellow_seconds:
                    state.in_yellow = False
                    state.phase_age = 0
                else:
                    state.phase_age += 1

            demand = sum(value for key, value in state.demand_profile.items() if key != "intersection_id")
            inbound = max(0.0, demand * self.random.uniform(0.05, 0.12))
            service_rate = state.outgoing_capacity * (1.0 + 0.2 * (state.current_phase % 2))
            discharge = min(state.queue_length + inbound, service_rate)
            spillback_penalty = 0.15 * max(0.0, state.queue_length - state.outgoing_capacity)
            state.queue_length = max(0.0, state.queue_length + inbound - discharge - spillback_penalty)
            state.occupancy_m2 = self._estimate_occupancy(state.queue_length, state.demand_profile)
            state.pressure = max(0.0, state.queue_length - state.outgoing_capacity)
            state.phase_age += 1
            state.phase_history.append(state.current_phase)

        self._update_neighbor_inference()
        return self.states

    def _estimate_occupancy(self, queue_length: float, demand_profile: Dict[str, float]) -> float:
        total_demand = max(1.0, sum(value for key, value in demand_profile.items() if key != "intersection_id"))
        weighted_area = 0.0
        for vehicle_class, share in self.calibration.vehicle_split.items():
            weighted_area += share * self.calibration.occupancy_m2_per_vehicle[vehicle_class]
        return queue_length * weighted_area * (1.0 + total_demand / 1000.0)

    def _update_neighbor_inference(self) -> None:
        for intersection_id, state in self.states.items():
            neighbor_ids = self.neighbors.get(intersection_id, ())
            if not neighbor_ids:
                state.neighbor_inference = {"avg_outflow": 0.0, "avg_pressure": 0.0}
                continue
            avg_outflow = float(np.mean([self.states[neighbor_id].outgoing_capacity for neighbor_id in neighbor_ids]))
            avg_pressure = float(np.mean([self.states[neighbor_id].pressure for neighbor_id in neighbor_ids]))
            state.neighbor_inference = {
                "avg_outflow": avg_outflow,
                "avg_pressure": avg_pressure,
            }


class NeuralGridChennaiEnv:
    def __init__(
        self,
        intersection_ids: Optional[Sequence[str]] = None,
        calibration: Optional[CorridorCalibration] = None,
        log_path: Optional[str] = None,
        seed: int = 7,
        use_sumo: bool = False,
    ) -> None:
        self.intersection_ids = list(
            intersection_ids
            or [
                "anna_salai_0",
                "anna_salai_1",
                "anna_salai_2",
                "anna_salai_3",
                "anna_salai_4",
            ]
        )
        self.calibration = calibration or CorridorCalibration()
        self.logger = DecisionAuditLogger(log_path=log_path)
        self.random = random.Random(seed)
        self.use_sumo = use_sumo and sumo_backend is not None and traci is not None
        self.backend = MockSumoBackend(self.intersection_ids, self.calibration, seed=seed)
        self.timestep = 0
        self.action_space_n = len(self.calibration.phase_plan)
        self.state_dim = 8

    @property
    def neighbors(self) -> Dict[str, Tuple[str, ...]]:
        return self.backend.neighbors

    def reset(self) -> Dict[str, np.ndarray]:
        states = self.backend.reset()
        self.timestep = 0
        return {intersection_id: self._build_observation(state) for intersection_id, state in states.items()}

    def _build_observation(self, state: IntersectionState) -> np.ndarray:
        inferred_pressure = state.neighbor_inference.get("avg_pressure", state.pressure)
        inferred_outflow = state.neighbor_inference.get("avg_outflow", state.outgoing_capacity)
        blind_flag = 1.0 if state.sensor_blind else 0.0
        phase_age_norm = min(1.0, state.phase_age / max(1, self.calibration.phase_plan[state.current_phase][1]))
        phase_index_norm = state.current_phase / max(1, self.action_space_n - 1)
        return np.asarray(
            [
                state.queue_length,
                state.occupancy_m2,
                state.incoming_capacity,
                state.outgoing_capacity,
                state.pressure,
                inferred_pressure,
                inferred_outflow,
                phase_age_norm + phase_index_norm + blind_flag,
            ],
            dtype=np.float32,
        )

    def get_state(self) -> Dict[str, np.ndarray]:
        return {intersection_id: self._build_observation(state) for intersection_id, state in self.backend.states.items()}

    def apply_data_dropout(self, states: Dict[str, np.ndarray], dropout_rate: float = 0.2) -> Dict[str, np.ndarray]:
        blinded_states: Dict[str, np.ndarray] = {}
        for intersection_id, observation in states.items():
            if self.random.random() < dropout_rate:
                blind_observation = observation.copy()
                blind_observation[:5] = 0.0
                blinded_states[intersection_id] = blind_observation
                self.backend.states[intersection_id].sensor_blind = True
            else:
                blinded_states[intersection_id] = observation
                self.backend.states[intersection_id].sensor_blind = False
        return blinded_states

    def infer_blind_state(self, intersection_id: str) -> np.ndarray:
        state = self.backend.states[intersection_id]
        neighbor_ids = self.neighbors.get(intersection_id, ())
        if not neighbor_ids:
            return self._build_observation(state)
        neighbor_outflow = float(np.mean([self.backend.states[neighbor_id].outgoing_capacity for neighbor_id in neighbor_ids]))
        neighbor_pressure = float(np.mean([self.backend.states[neighbor_id].pressure for neighbor_id in neighbor_ids]))
        inferred = self._build_observation(state).copy()
        inferred[5] = neighbor_pressure
        inferred[6] = neighbor_outflow
        return inferred

    def safety_mask(self, intersection_id: str, proposed_action: int) -> int:
        state = self.backend.states[intersection_id]
        if state.phase_age < self.calibration.min_green_seconds:
            return state.current_phase
        if proposed_action < 0 or proposed_action >= self.action_space_n:
            return state.current_phase
        return proposed_action

    def _compute_reward(self, state: IntersectionState) -> float:
        delay = state.queue_length + 0.5 * state.pressure
        pressure_reward = -state.pressure
        stability_bonus = 1.0 if not state.in_yellow else -1.0
        return float(pressure_reward - 0.1 * delay + stability_bonus)

    def step(
        self,
        actions: Dict[str, int],
        log_decisions: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        safe_actions = {
            intersection_id: self.safety_mask(intersection_id, action)
            for intersection_id, action in actions.items()
        }
        states = self.backend.step(safe_actions)
        self.timestep += 1

        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        dones: Dict[str, bool] = {}
        infos: Dict[str, dict] = {}

        for intersection_id, state in states.items():
            observation = self._build_observation(state)
            observations[intersection_id] = observation
            rewards[intersection_id] = self._compute_reward(state)
            dones[intersection_id] = False
            infos[intersection_id] = {
                "pressure": state.pressure,
                "neighbor_inference": state.neighbor_inference,
                "occupancy_m2": state.occupancy_m2,
                "queue_length": state.queue_length,
            }
            if log_decisions:
                self.logger.log(
                    DecisionLogEntry(
                        timestep=self.timestep,
                        intersection_id=intersection_id,
                        action=int(safe_actions.get(intersection_id, state.current_phase)),
                        pressure=state.pressure,
                        inferred_neighbor_state=state.neighbor_inference,
                        state_vector=observation.tolist(),
                    )
                )

        return observations, rewards, dones, infos

    def global_pressure(self) -> float:
        return float(np.mean([state.pressure for state in self.backend.states.values()]))

    def metrics_snapshot(self) -> Dict[str, dict]:
        snapshot: Dict[str, dict] = {}
        for intersection_id, state in self.backend.states.items():
            snapshot[intersection_id] = {
                "pressure": state.pressure,
                "queue_length": state.queue_length,
                "occupancy_m2": state.occupancy_m2,
                "current_phase": state.current_phase,
                "neighbor_inference": state.neighbor_inference,
                "blind": state.sensor_blind,
            }
        snapshot["global_pressure"] = {"value": self.global_pressure()}
        return snapshot


def build_default_environment(log_path: Optional[str] = None) -> NeuralGridChennaiEnv:
    return NeuralGridChennaiEnv(log_path=log_path)