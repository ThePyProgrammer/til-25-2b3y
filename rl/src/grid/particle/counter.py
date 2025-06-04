from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..utils import Point


class VisitCounter:
    def __init__(self, size: int = 16):
        self.size = size
        self.counts: NDArray[np.float32] = np.zeros((size, size), dtype=np.float32)

        self._cached_counts: Optional[NDArray[np.float32]] = None
        self._cached_nonzero: Optional[list[Point]] = None

    def update(self, other: 'VisitCounter'):
        self.counts += other.counts

    def __getitem__(self, position: Point) -> int:
        return self.counts[position.x, position.y] # type: ignore

    def __setitem__(self, position: Point, count: int):
        self.counts[position.x, position.y] = count

    def __repr__(self) -> str:
        return str(self.counts)

    def copy(self) -> 'VisitCounter':
        new_counter = VisitCounter(size=self.size)
        new_counter.counts = self.counts.copy()

        return new_counter

    def nonzero(self) -> list[Point]:
        if self._cached_counts is None or self._cached_nonzero is None or not np.array_equal(self.counts, self._cached_counts):
            self._cached_counts = self.counts.copy()  # Make a copy, not a reference
            self._cached_nonzero = [Point(x, y) for x, y in zip(*np.nonzero(self.counts))]
        return self._cached_nonzero

    def scale(self, factor: float):
        self.counts = self.counts * factor
