import numpy as np
from numpy.typing import NDArray

from ..utils import Point

class VisitCounter:
    def __init__(self, size: int = 16):
        self.size = size
        self.counts: NDArray[np.uint32] = np.zeros((size, size), dtype=np.uint32)

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
        new_counter.counts = self.counts

        return new_counter

    def nonzero(self) -> list[Point]:
        return [Point(x, y) for x, y in zip(*np.nonzero(self.counts))]
