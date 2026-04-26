from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from agent_logic import D3QNAgent
from environment import NeuralGridChennaiEnv, build_default_environment


class FixedTimeBaseline:
    def __init__(self, action_dim: int, cycle_length: int = 4) -> None:
        self.action_dim = action_dim
        self.cycle_length = cycle_length
        self.step_count = 0

    def act(self, observation: np.ndarray) -> int:
        action = (self.step_count // self.cycle_length) % self.action_dim
        self.step_count += 1
        return action


def evaluate_policy(env: NeuralGridChennaiEnv, agent: D3QNAgent, horizon: int = 200, fixed_time: bool = False) -> Dict[str, float]:
    observations = env.reset()
    baseline = FixedTimeBaseline(env.action_space_n)
    cumulative_reward = 0.0
    cumulative_pressure = 0.0
    cumulative_delay = 0.0

    for _ in range(horizon):
        actions = {}
        for intersection_id, observation in observations.items():
            if fixed_time:
                actions[intersection_id] = baseline.act(observation)
            else:
                actions[intersection_id] = agent.select_action(observation, exploit=True)
        next_observations, rewards, dones, infos = env.step(actions, log_decisions=not fixed_time)
        cumulative_reward += sum(rewards.values())
        cumulative_pressure += sum(info["pressure"] for info in infos.values())
        cumulative_delay += sum(info["queue_length"] for info in infos.values())
        observations = next_observations

    denominator = max(1, horizon * len(env.intersection_ids))
    return {
        "reward": cumulative_reward / denominator,
        "pressure": cumulative_pressure / denominator,
        "delay": cumulative_delay / denominator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Neural-Grid Chennai against a fixed-time baseline.")
    parser.add_argument("--model-path", type=str, default="models/d3qn_chennai.pt")
    parser.add_argument("--horizon", type=int, default=200)
    args = parser.parse_args()

    env = build_default_environment()
    agent = D3QNAgent(state_dim=env.state_dim, action_dim=env.action_space_n)
    model_path = Path(args.model_path)
    if model_path.exists():
        agent.load(str(model_path))

    rl_metrics = evaluate_policy(env, agent, horizon=args.horizon, fixed_time=False)
    baseline_env = build_default_environment()
    baseline_metrics = evaluate_policy(baseline_env, agent, horizon=args.horizon, fixed_time=True)

    improvement = baseline_metrics["delay"] - rl_metrics["delay"]
    print("RL metrics:", rl_metrics)
    print("Fixed-time metrics:", baseline_metrics)
    print(f"delay_reduction={improvement:.4f}")


if __name__ == "__main__":
    main()