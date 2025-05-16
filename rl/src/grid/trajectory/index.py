from typing import Any

from .trajectory import Trajectory


class TrajectoryIndex:
    def __init__(self):
        self._index: dict[Any, set[Trajectory]] = {}

    def add(self, key: Any, trajectory: Trajectory) -> None:
        if key not in self._index:
            self._index[key] = set()
        self._index[key].add(trajectory)

    def remove(self, key: Any, trajectory: Trajectory) -> None:
        if key in self._index and trajectory in self._index[key]:
            trajectory.prune()
            self._index[key].remove(trajectory)
            # Clean up empty sets
            if not self._index[key]:
                del self._index[key]

    def clear(self):
        self._index.clear()

    def __contains__(self, key: Any):
        return key in self._index

    def __getitem__(self, key: Any):
        if key not in self._index:
            self._index[key] = set()
        return self._index[key]
