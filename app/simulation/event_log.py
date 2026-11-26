"""Immutable, strictly time-ordered event log store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from app.domain.enums import EventType
from app.domain.events import EventLogRecord


class EventLog:
    """Append-only simulation event log enforcing monotonic timestamp & sequence ordering."""

    def __init__(self, run_id: str, scenario_id: str, seed: int, synthetic: bool = True) -> None:
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.seed = seed
        self.synthetic = synthetic
        self._records: list[EventLogRecord] = []
        self._last_timestamp: float = 0.0

    def append(
        self,
        timestamp_minute: float,
        event_type: EventType,
        entity_id: str,
        entity_type: str,
        location_id: str | None = None,
        order_id: str | None = None,
        resource_id: str | None = None,
        payload: dict[str, Any] | None = None,
        available_at_minute: float | None = None,
    ) -> EventLogRecord:
        """Append an immutable event after validating causal timestamp monotonicity."""
        ts = float(timestamp_minute)
        if ts < self._last_timestamp - 1e-6:
            raise ValueError(
                f"Non-monotonic event timestamp: {ts} < last {self._last_timestamp} for {event_type}"
            )
        ts = max(ts, self._last_timestamp)
        self._last_timestamp = ts
        rounded_ts = round(ts, 4)
        avail = round(float(available_at_minute), 4) if available_at_minute is not None else rounded_ts
        record = EventLogRecord(
            sequence_no=len(self._records),
            timestamp_minute=rounded_ts,
            available_at_minute=max(rounded_ts, avail),
            event_type=event_type,
            entity_id=entity_id,
            entity_type=entity_type,
            location_id=location_id,
            order_id=order_id,
            resource_id=resource_id,
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            seed=self.seed,
            synthetic=self.synthetic,
            payload=payload or {},
        )
        self._records.append(record)
        return record

    @property
    def records(self) -> tuple[EventLogRecord, ...]:
        """Return immutable tuple view of all logged events."""
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def to_polars(self) -> pl.DataFrame:
        """Convert event records into a Polars DataFrame suitable for Parquet export."""
        import json

        if not self._records:
            return pl.DataFrame(
                schema={
                    "event_id": pl.Utf8,
                    "sequence_no": pl.Int64,
                    "timestamp_minute": pl.Float64,
                    "event_type": pl.Utf8,
                    "entity_id": pl.Utf8,
                    "entity_type": pl.Utf8,
                    "location_id": pl.Utf8,
                    "order_id": pl.Utf8,
                    "resource_id": pl.Utf8,
                    "run_id": pl.Utf8,
                    "scenario_id": pl.Utf8,
                    "seed": pl.Int32,
                    "payload_json": pl.Utf8,
                }
            )
        rows = [
            {
                "event_id": r.event_id,
                "sequence_no": r.sequence_no,
                "timestamp_minute": r.timestamp_minute,
                "event_type": str(r.event_type),
                "entity_id": r.entity_id,
                "entity_type": r.entity_type,
                "location_id": r.location_id,
                "order_id": r.order_id,
                "resource_id": r.resource_id,
                "run_id": r.run_id,
                "scenario_id": r.scenario_id,
                "seed": r.seed,
                "payload_json": json.dumps(r.payload, sort_keys=True, default=str),
            }
            for r in self._records
        ]
        return pl.DataFrame(rows)

    def to_parquet(self, path: str | Path) -> Path:
        """Write event log to Parquet file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_polars().write_parquet(target)
        return target
