"""Immutable, strictly time-ordered event schema for simulation logging."""

from __future__ import annotations

from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import EventType


class EventLogRecord(BaseModel):
    """Immutable event emitted by the discrete-event simulator with disclosure timestamp."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    sequence_no: int = Field(ge=0)
    timestamp_minute: float = Field(ge=0)
    available_at_minute: float | None = Field(default=None, ge=0)
    event_type: EventType
    entity_id: str
    entity_type: str
    location_id: str | None = None
    order_id: str | None = None
    resource_id: str | None = None
    run_id: str
    scenario_id: str
    seed: int
    synthetic: bool = True
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_disclosure_time(self) -> Self:
        """Ensure disclosure time never precedes occurrence time."""
        if self.available_at_minute is None:
            object.__setattr__(self, "available_at_minute", self.timestamp_minute)
        elif self.available_at_minute < self.timestamp_minute:
            raise ValueError("available_at_minute precedes timestamp_minute")
        return self
