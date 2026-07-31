"""Temporal aggregation for chunked robot action predictions."""

from __future__ import annotations

import math

import numpy as np


class TemporalEnsemble:
    """Aggregate overlapping action chunks while treating the gripper discretely."""

    MODES = {
        "average",
        "confirm",
        "confirm_then_hold",
        "debounce",
        "hold",
        "hysteresis",
        "latest",
        "latch",
    }

    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        decay: float = 0.5,
        *,
        gripper_mode: str = "latest",
        gripper_close_threshold: float = 0.5,
        gripper_confirm_steps: int = 2,
        gripper_hold_steps: int = 4,
    ) -> None:
        if chunk_size < 1 or action_dim < 1:
            raise ValueError("chunk_size and action_dim must be positive")
        if not math.isfinite(decay) or not 0 < decay <= 1:
            raise ValueError("decay must be finite and in (0, 1]")
        if gripper_mode not in self.MODES:
            raise ValueError(f"gripper_mode must be one of {sorted(self.MODES)}")
        if not math.isfinite(gripper_close_threshold) or not -1 <= gripper_close_threshold <= 1:
            raise ValueError("gripper_close_threshold must be finite and in [-1, 1]")
        if gripper_confirm_steps < 1:
            raise ValueError("gripper_confirm_steps must be positive")
        if gripper_hold_steps < 1:
            raise ValueError("gripper_hold_steps must be positive")
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay
        self.gripper_mode = gripper_mode
        self.gripper_close_threshold = gripper_close_threshold
        self.gripper_confirm_steps = gripper_confirm_steps
        self.gripper_hold_steps = gripper_hold_steps
        self.reset()

    def reset(self) -> None:
        """Discard stale chunks and discrete gripper state after a controller retry."""
        self._buffer: dict[int, list[tuple[np.ndarray, float]]] = {}
        self._gripper_latched = False
        self._gripper_close_streak = 0
        self._gripper_hold_remaining = 0
        self._gripper_confirmed = False
        self.last_raw_gripper: float | None = None

    def add_chunk(self, start_step: int, chunk: np.ndarray) -> None:
        values = np.asarray(chunk)
        if values.ndim != 2 or values.shape[1] != self.action_dim or len(values) < 1:
            raise ValueError(
                f"Expected a non-empty [time, {self.action_dim}] chunk; got {values.shape}"
            )
        weights = self.decay ** np.arange(len(values))
        for index, (action, weight) in enumerate(zip(values, weights)):
            self._buffer.setdefault(start_step + index, []).append((action.copy(), float(weight)))

    def get_action(self, step: int) -> np.ndarray:
        entries = self._buffer.pop(step, None)
        if entries is None:
            raise ValueError(f"No prediction for step {step}")
        actions, raw_weights = zip(*entries)
        weights = np.asarray(raw_weights, dtype=float)
        averaged = np.average(actions, axis=0, weights=weights)

        if self.action_dim >= 7 and self.gripper_mode != "average":
            latest = float(actions[-1][6])
            self.last_raw_gripper = latest
            if self.gripper_mode == "hold":
                if latest > self.gripper_close_threshold:
                    self._gripper_hold_remaining = self.gripper_hold_steps
                if self._gripper_hold_remaining > 0:
                    averaged[6] = 1.0
                    self._gripper_hold_remaining -= 1
                else:
                    averaged[6] = latest
            elif self.gripper_mode == "confirm_then_hold":
                if latest > self.gripper_close_threshold:
                    self._gripper_close_streak += 1
                else:
                    self._gripper_close_streak = 0

                if not self._gripper_confirmed:
                    self._gripper_confirmed = (
                        self._gripper_close_streak >= self.gripper_confirm_steps
                    )
                    if self._gripper_confirmed:
                        self._gripper_hold_remaining = self.gripper_hold_steps

                if self._gripper_confirmed and self._gripper_hold_remaining > 0:
                    averaged[6] = 1.0
                    self._gripper_hold_remaining -= 1
                elif self._gripper_confirmed and latest > self.gripper_close_threshold:
                    averaged[6] = 1.0
                elif not self._gripper_confirmed:
                    averaged[6] = -1.0
                else:
                    # A close that survived the minimum hold can recover from
                    # an early false positive; unlike debounce this is not a
                    # permanent latch.
                    self._gripper_confirmed = False
                    averaged[6] = latest
            elif self.gripper_mode in {"confirm", "debounce", "hysteresis"}:
                if latest > self.gripper_close_threshold:
                    self._gripper_close_streak += 1
                else:
                    self._gripper_close_streak = 0
                if self.gripper_mode == "confirm":
                    averaged[6] = (
                        1.0
                        if self._gripper_close_streak >= self.gripper_confirm_steps
                        else -1.0
                    )
                    return averaged
                self._gripper_latched = self._gripper_latched or (
                    self._gripper_close_streak >= self.gripper_confirm_steps
                )
                if self._gripper_latched:
                    averaged[6] = 1.0
                elif self.gripper_mode == "debounce":
                    averaged[6] = -1.0
                else:
                    averaged[6] = latest
            elif self.gripper_mode == "latch":
                self._gripper_latched = self._gripper_latched or (
                    latest > self.gripper_close_threshold
                )
                averaged[6] = 1.0 if self._gripper_latched else latest
            else:
                averaged[6] = latest
        return averaged

    def has_action(self, step: int) -> bool:
        return step in self._buffer
