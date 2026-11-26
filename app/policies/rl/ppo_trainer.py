"""Stable-Baselines3 PPO training pipeline for reverse-logistics dispatching."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from app.config import RLSettings, SimulationSettings
from app.domain.entities import NetworkDataset, Scenario
from app.simulation.env import ReverseLogisticsEnv


class MultiScenarioTrainEnv(gym.Env[np.ndarray, np.ndarray]):
    """Wrapper cycling deterministically across training scenarios & seeds per episode."""

    def __init__(
        self,
        dataset: NetworkDataset,
        scenarios: Sequence[Scenario],
        sim_settings: SimulationSettings,
        base_seed: int = 42,
    ) -> None:
        super().__init__()
        self.scenarios = list(scenarios)
        self.base_seed = base_seed
        self._episode_idx = 0
        self.inner = ReverseLogisticsEnv(
            dataset=dataset,
            scenario=self.scenarios[0],
            sim_settings=sim_settings,
            policy_id="ppo_training",
            policy_type="rl",
            seed=base_seed,
        )
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        scenario = self.scenarios[self._episode_idx % len(self.scenarios)]
        ep_seed = (seed if seed is not None else self.base_seed) + self._episode_idx * 17
        self._episode_idx += 1
        merged_opts = dict(options or {})
        merged_opts["scenario"] = scenario
        return self.inner.reset(seed=ep_seed, options=merged_opts)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.inner.step(action)


def train_ppo_policy(
    dataset: NetworkDataset,
    scenarios: Sequence[Scenario],
    rl_settings: RLSettings | None = None,
    sim_settings: SimulationSettings | None = None,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train a compact MlpPolicy PPO agent and save model artifact."""
    cfg = rl_settings or RLSettings()
    sim_cfg = sim_settings or SimulationSettings()
    target = Path(output_path or cfg.model_output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    def _make_env() -> gym.Env[np.ndarray, np.ndarray]:
        env = MultiScenarioTrainEnv(
            dataset=dataset,
            scenarios=scenarios,
            sim_settings=sim_cfg,
            base_seed=seed,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(str(target))
    elapsed = time.perf_counter() - t0
    vec_env.close()

    return {
        "model_path": str(target),
        "total_timesteps": cfg.total_timesteps,
        "seed": seed,
        "training_seconds": round(elapsed, 2),
        "net_arch": list(cfg.net_arch),
    }
