"""Tests for Gymnasium env compliance and Stable-Baselines3 PPO training & inference."""

from __future__ import annotations

from gymnasium.utils.env_checker import check_env

from app.experiments.runner import run_single_episode
from app.policies.rl.ppo_policy import RLPPOPolicy
from app.policies.rl.ppo_trainer import train_ppo_policy
from app.simulation.env import ReverseLogisticsEnv


def test_gymnasium_check_env(sample_dataset, baseline_scenario, test_settings) -> None:
    env = ReverseLogisticsEnv(
        dataset=sample_dataset,
        scenario=baseline_scenario,
        sim_settings=test_settings.simulation,
        seed=42,
    )
    check_env(env, skip_render_check=True)


def test_ppo_train_and_backtest_roundtrip(sample_dataset, baseline_scenario, test_settings) -> None:
    train_res = train_ppo_policy(
        dataset=sample_dataset,
        scenarios=[baseline_scenario],
        rl_settings=test_settings.rl,
        sim_settings=test_settings.simulation,
        seed=42,
        output_path=test_settings.rl.model_output_path,
    )
    assert train_res["total_timesteps"] == 512

    ppo = RLPPOPolicy(model_path=test_settings.rl.model_output_path)
    run = run_single_episode(
        dataset=sample_dataset,
        scenario=baseline_scenario,
        policy=ppo,
        seed=101,
        settings=test_settings,
        experiment_name="test_ppo",
    )
    assert run.status == "completed"
    assert run.metrics.total_orders > 0
