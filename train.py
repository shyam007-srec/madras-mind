from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from agent_logic import D3QNAgent
from environment import NeuralGridChennaiEnv, build_default_environment


@dataclass
class TrainingConfig:
    episodes: int = 50
    horizon: int = 200
    batch_size: int = 64
    dropout_rate: float = 0.2
    save_path: str = "models/d3qn_chennai.pt"
    log_path: str = "models/training_log.jsonl"


class AdversarialNoiseInjector:
    def __init__(self, dropout_rate: float = 0.2, seed: int = 11) -> None:
        self.dropout_rate = dropout_rate
        self.random = np.random.default_rng(seed)

    def apply(self, observation: np.ndarray) -> np.ndarray:
        noisy_observation = observation.copy()
        if self.random.random() < self.dropout_rate:
            noisy_observation[:5] = 0.0
        return noisy_observation


def run_episode(env: NeuralGridChennaiEnv, agent: D3QNAgent, config: TrainingConfig, train: bool = True) -> Dict[str, float]:
    observations = env.reset()
    injector = AdversarialNoiseInjector(dropout_rate=config.dropout_rate)
    cumulative_reward = 0.0
    cumulative_pressure = 0.0

    for _ in range(config.horizon):
        current_actions: Dict[str, int] = {}
        processed_observations: Dict[str, np.ndarray] = {}
        for intersection_id, observation in observations.items():
            inference_source = env.infer_blind_state(intersection_id)
            mixed_observation = observation.copy()
            if env.backend.states[intersection_id].sensor_blind:
                mixed_observation = inference_source
            mixed_observation = injector.apply(mixed_observation)
            processed_observations[intersection_id] = mixed_observation
            current_actions[intersection_id] = agent.select_action(mixed_observation, exploit=not train)

        next_observations, rewards, dones, infos = env.step(current_actions, log_decisions=True)
        for intersection_id in env.intersection_ids:
            agent.store(
                processed_observations[intersection_id],
                current_actions[intersection_id],
                rewards[intersection_id],
                next_observations[intersection_id],
                False,
            )
            cumulative_reward += rewards[intersection_id]
            cumulative_pressure += infos[intersection_id]["pressure"]
        if train:
            agent.update(config.batch_size)
        observations = next_observations

    return {
        "reward": cumulative_reward,
        "pressure": cumulative_pressure / max(1, config.horizon * len(env.intersection_ids)),
    }


def train() -> None:
    parser = argparse.ArgumentParser(description="Train the Neural-Grid Chennai D3QN agent.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--save-path", type=str, default="models/d3qn_chennai.pt")
    parser.add_argument("--log-path", type=str, default="models/training_log.jsonl")
    args = parser.parse_args()

    config = TrainingConfig(
        episodes=args.episodes,
        horizon=args.horizon,
        save_path=args.save_path,
        log_path=args.log_path,
    )
    Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.log_path).parent.mkdir(parents=True, exist_ok=True)

    env = build_default_environment(log_path=config.log_path)
    agent = D3QNAgent(state_dim=env.state_dim, action_dim=env.action_space_n)

    summary: List[Dict[str, float]] = []
    for episode in range(1, config.episodes + 1):
        metrics = run_episode(env, agent, config, train=True)
        summary.append(metrics)
        with Path(config.log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"episode": episode, **metrics}) + "\n")
        print(f"episode={episode} reward={metrics['reward']:.3f} pressure={metrics['pressure']:.3f} epsilon={agent.epsilon:.3f}")

    agent.save(config.save_path)
    print(f"saved_model={config.save_path}")


if __name__ == "__main__":
    train()