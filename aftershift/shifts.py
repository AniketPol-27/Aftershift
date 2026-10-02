"""Shift events and schedules for mid-episode dynamics shifts.

A `ShiftScheduler` mutates an env's context at prescribed step indices. Convention:
`apply(env, t)` is called BEFORE env step `t` executes, so an event at `t` affects the
transition from step t to t+1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ShiftEvent:
    t: int          # step index at which the shift occurs
    params: dict    # context parameters to set at that step


def step_events(t_shift: int, **params) -> list[ShiftEvent]:
    """Single abrupt shift at step `t_shift`."""
    return [ShiftEvent(t_shift, dict(params))]


def recurring_events(t_start: int, period: int, n: int, **params) -> list[ShiftEvent]:
    """The same shift applied n times every `period` steps."""
    return [ShiftEvent(t_start + i * period, dict(params)) for i in range(n)]


def ramp_events(t_start: int, t_end: int, n: int, param: str, mult_from: float,
                mult_to: float, nominal_value: float) -> list[ShiftEvent]:
    """Gradual drift of one parameter from mult_from to mult_to (times nominal)."""
    ts = np.linspace(t_start, t_end, n)
    evs = []
    for i, t in enumerate(ts):
        mult = mult_from + (mult_to - mult_from) * i / max(n - 1, 1)
        evs.append(ShiftEvent(int(t), {param: nominal_value * mult}))
    return evs


class ShiftScheduler:
    """Applies a list of ShiftEvents to an env at the right steps (per episode)."""

    def __init__(self, events: Iterable[ShiftEvent]):
        self.events = sorted(events, key=lambda e: e.t)
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def apply(self, env, t: int) -> None:
        while self._i < len(self.events) and self.events[self._i].t == t:
            env.set_context(**self.events[self._i].params)
            self._i += 1
