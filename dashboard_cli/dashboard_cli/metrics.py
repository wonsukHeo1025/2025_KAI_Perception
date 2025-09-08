from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional
import time


@dataclass
class TopicStats:
    window_sec: float = 5.0
    times: Deque[float] = field(default_factory=lambda: deque(maxlen=256))
    last_data: Optional[object] = None

    def add_event(self, data: object = None, t: Optional[float] = None) -> None:
        if t is None:
            t = time.time()
        self.times.append(t)
        self.last_data = data

    @property
    def hz(self) -> float:
        if len(self.times) < 2:
            return 0.0
        t_now = self.times[-1]
        # consider only events within window
        t0 = t_now - self.window_sec
        vals = [x for x in self.times if x >= t0]
        if len(vals) < 2:
            return 0.0
        span = vals[-1] - vals[0]
        if span <= 0:
            return 0.0
        return (len(vals) - 1) / span

    @property
    def age(self) -> float:
        if not self.times:
            return float('inf')
        return max(0.0, time.time() - self.times[-1])

