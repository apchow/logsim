"""Stable-Baselines3 PPO inference policy adapter for Gymnasium backtesting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from app.domain.enums import PolicyType
from app.policies.base import BasePolicy


class RLPPOPolicy(BasePolicy):
    """Trained Stable-Baselines3 PPO policy operating strictly on causal observations."""

    policy_id = "ppo_agent"
    policy_type = PolicyType.RL
    policy_version = "1.0.0"

    def __init__(
        self,
        model_path: str | Path = "artifacts/models/ppo_reverse_dispatcher.zip",
        deterministic: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self._model: PPO | None = None

    def _ensure_loaded(self) -> PPO:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Trained PPO model not found at '{self.model_path}'. "
                    "Run `logisticsbt train` first or enable auto-training."
                )
            self._model = PPO.load(str(self.model_path), device="cpu")
        return self._model

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        _ = info
        model = self._ensure_loaded()
        action, _ = model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.int64).reshape(-1)
