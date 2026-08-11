from __future__ import annotations

from collections import deque


class BCProgressGate:
    """Simple piecewise gate for scaling BC strength by recent success."""

    def __init__(
        self,
        *,
        thresholds: tuple[float, float, float] = (0.1, 0.4, 0.7),
        multipliers: tuple[float, float, float, float] = (0.0, 0.1, 0.4, 1.0),
        window: int = 3,
    ) -> None:
        low, mid, high = thresholds
        if not (0.0 <= low < mid < high <= 1.0):
            raise ValueError("thresholds must satisfy 0 <= low < mid < high <= 1.")
        if window <= 0:
            raise ValueError("window must be positive.")
        if len(multipliers) != 4 or any(
            value < 0.0 or value > 1.0 for value in multipliers
        ):
            raise ValueError("multipliers must contain four values in [0, 1].")
        self.thresholds = thresholds
        self.multipliers = multipliers
        self.window = int(window)
        self._history: deque[float] = deque(maxlen=self.window)
        self.score = 0.0
        self.multiplier = float(multipliers[0])

    def reset(self) -> None:
        self._history.clear()
        self.score = 0.0
        self.multiplier = float(self.multipliers[0])

    def update(self, success: float) -> float:
        self._history.append(float(success))
        self.score = sum(self._history) / len(self._history)
        low, mid, high = self.thresholds
        early, low_mult, mid_mult, high_mult = self.multipliers
        if self.score < low:
            self.multiplier = float(early)
        elif self.score < mid:
            self.multiplier = float(low_mult)
        elif self.score < high:
            self.multiplier = float(mid_mult)
        else:
            self.multiplier = float(high_mult)
        return self.multiplier
