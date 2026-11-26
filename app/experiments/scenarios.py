"""Versioned scenario registry supporting YAML bundles and individual scenario files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import load_yaml_config
from app.domain.entities import Scenario, ScenarioOverlay
from app.domain.enums import OverlayType


def default_scenarios() -> list[Scenario]:
    """Return built-in versioned reverse-logistics scenarios."""
    return [
        Scenario(
            scenario_id="baseline_normal",
            version="1.0.0",
            name="Baseline Reverse Flow",
            description="Normal 3-day reverse parcel pickup and hub triage operations.",
            horizon_minutes=4320.0,
            decision_interval_minutes=60.0,
            overlays=[],
            is_synthetic=True,
        )
    ]


def _coerce_scenario_item(raw_item: dict[str, Any]) -> Scenario:
    """Parse either structured Scenario schema or flat Kimi-style scenario YAML overlays."""
    overlays_raw = raw_item.get("overlays", [])
    if isinstance(overlays_raw, list):
        return Scenario.model_validate(raw_item)

    overlays: list[ScenarioOverlay] = []
    if isinstance(overlays_raw, dict):
        if "demand_multiplier" in overlays_raw and float(overlays_raw["demand_multiplier"]) != 1.0:
            overlays.append(
                ScenarioOverlay(
                    overlay_type=OverlayType.DEMAND_MULTIPLIER,
                    start_minute=0.0,
                    value=float(overlays_raw["demand_multiplier"]),
                )
            )
        if "demand_mix_shift" in overlays_raw and float(overlays_raw["demand_mix_shift"]) > 0.0:
            overlays.append(
                ScenarioOverlay(
                    overlay_type=OverlayType.DEMAND_MIX_SHIFT,
                    start_minute=0.0,
                    value=float(overlays_raw["demand_mix_shift"]),
                )
            )
        if "carrier_delay_minutes" in overlays_raw and float(overlays_raw["carrier_delay_minutes"]) > 0:
            overlays.append(
                ScenarioOverlay(
                    overlay_type=OverlayType.CARRIER_DELAY,
                    start_minute=300.0,
                    end_minute=3000.0,
                    value=1.0 + float(overlays_raw["carrier_delay_minutes"]) / 100.0,
                )
            )
        if "lead_time_increase_pct" in overlays_raw and float(overlays_raw["lead_time_increase_pct"]) > 0:
            overlays.append(
                ScenarioOverlay(
                    overlay_type=OverlayType.LEAD_TIME_INCREASE,
                    start_minute=300.0,
                    end_minute=3000.0,
                    value=1.0 + float(overlays_raw["lead_time_increase_pct"]) / 100.0,
                )
            )

    return Scenario(
        scenario_id=str(raw_item["scenario_id"]),
        version=str(raw_item.get("version", "1.0.0")),
        name=str(raw_item.get("name", raw_item["scenario_id"])),
        description=str(raw_item.get("description", "")),
        horizon_minutes=float(raw_item.get("horizon_minutes", 4320.0)),
        decision_interval_minutes=float(raw_item.get("decision_interval_minutes", 60.0)),
        overlays=overlays,
        is_synthetic=bool(raw_item.get("is_synthetic", True)),
    )


def load_scenarios(path: str | Path = "configs/scenarios.yaml") -> dict[str, Scenario]:
    """Load versioned scenarios from YAML config and optional configs/scenarios/*.yaml files."""
    scenarios_map: dict[str, Scenario] = {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        pkg_fallback = Path(__file__).resolve().parent.parent.parent / "configs" / "scenarios.yaml"
        if pkg_fallback.exists():
            cfg_path = pkg_fallback

    if cfg_path.exists():
        raw = load_yaml_config(cfg_path)
        for item in raw.get("scenarios", []):
            s = _coerce_scenario_item(item)
            scenarios_map[s.scenario_id] = s

    # Load scenario YAML files from both package/config dir and current working dir (configs/scenarios/)
    candidate_dirs: list[Path] = []
    for cand in (cfg_path.parent / "scenarios", Path("configs/scenarios")):
        if cand.is_dir() and cand.resolve() not in {d.resolve() for d in candidate_dirs}:
            candidate_dirs.append(cand)
    for scen_dir in candidate_dirs:
        for yaml_file in sorted(scen_dir.glob("*.yaml")):
            item = load_yaml_config(yaml_file)
            if "scenario_id" in item:
                s = _coerce_scenario_item(item)
                scenarios_map[s.scenario_id] = s

    if not scenarios_map:
        return {s.scenario_id: s for s in default_scenarios()}
    return scenarios_map
