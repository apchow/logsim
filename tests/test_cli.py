"""Typer CLI integration tests for all 7 logisticsbt commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from app.cli.main import app

runner = CliRunner()


def test_cli_ingest_validate_simulate_backtest_report_export(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    res_ingest = runner.invoke(
        app,
        ["ingest", "--synthetic", "--seed", "42", "--orders-per-day", "40", "--output", "data/syn"],
    )
    assert res_ingest.exit_code == 0, res_ingest.output

    res_val = runner.invoke(app, ["validate", "--dataset", "data/syn"])
    assert res_val.exit_code == 0, res_val.output

    res_sim = runner.invoke(
        app,
        ["simulate", "--policy", "fifo_edf", "--scenario", "baseline_normal", "--seed", "101", "--dataset", "data/syn", "--experiment", "cli_test"],
    )
    assert res_sim.exit_code == 0, res_sim.output

    res_train = runner.invoke(
        app,
        ["train", "--timesteps", "512", "--seed", "42", "--dataset", "data/syn", "--output", "artifacts/models/ppo_reverse_dispatcher.zip"],
    )
    assert res_train.exit_code == 0, res_train.output

    res_bt = runner.invoke(
        app,
        [
            "backtest",
            "--policies",
            "current_replay,fifo_edf,ppo_agent",
            "--scenarios",
            "baseline_normal",
            "--seeds",
            "101,202,303,404,505",
            "--dataset",
            "data/syn",
            "--experiment",
            "cli_bt",
        ],
    )
    assert res_bt.exit_code == 0, res_bt.output

    res_rep = runner.invoke(app, ["report", "--experiment", "cli_bt", "--baseline", "current_replay"])
    assert res_rep.exit_code == 0, res_rep.output

    res_exp = runner.invoke(app, ["export", "--experiment", "cli_bt", "--format", "csv", "--output", "out.csv"])
    assert res_exp.exit_code == 0, res_exp.output
    assert (tmp_path / "out.csv").exists()
