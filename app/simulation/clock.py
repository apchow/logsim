"""Simulation time, shift calendar, and HOS helper functions."""

from __future__ import annotations

from app.domain.entities import ResourceCalendar


def minute_of_day(sim_minute: float) -> float:
    """Return minute offset within the current 24h day [0, 1440)."""
    return sim_minute % 1440.0


def hour_of_day(sim_minute: float) -> float:
    """Return fractional hour of day [0, 24)."""
    return (sim_minute % 1440.0) / 60.0


def day_of_sim(sim_minute: float) -> int:
    """Return integer simulation day index starting at 0."""
    return int(sim_minute // 1440.0)


def next_shift_open_minute(calendar: ResourceCalendar, sim_minute: float) -> float:
    """Return the earliest simulation minute >= sim_minute when the calendar is active."""
    if calendar.is_active(sim_minute):
        return sim_minute
    cur_day = day_of_sim(sim_minute)
    for offset in range(8):
        candidate_day = cur_day + offset
        if (candidate_day % 7) not in calendar.working_days:
            continue
        candidate_min = candidate_day * 1440.0 + calendar.shift_start_hour * 60.0
        if candidate_min >= sim_minute:
            return candidate_min
    return sim_minute + 60.0
