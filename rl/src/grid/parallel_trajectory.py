import concurrent.futures
from typing import List, Dict, Tuple, Any, Optional

from .trajectory import TrajectoryTree, Trajectory


class ParallelTrajectoryTree(TrajectoryTree):
    """
    A multithreaded implementation of TrajectoryTree that can consider more
    trajectories per endpoint by parallelizing the expansion process.
    """

    def __init__(self, *args, num_threads: int = 8, max_trajectories_per_endpoint: int = 10, **kwargs):
        """
        Initialize the ParallelTrajectoryTree.

        Args:
            num_threads: Number of threads to use for parallel processing
            max_trajectories_per_endpoint: Maximum number of trajectories to consider per endpoint
            *args, **kwargs: Arguments passed to TrajectoryTree.__init__
        """
        super().__init__(*args, **kwargs)
        self.num_threads = num_threads
        self.max_trajectories_per_endpoint = max_trajectories_per_endpoint

    def step(self) -> int:
        """
        Multithreaded version of step that considers more edge trajectories.

        Expands trajectories in parallel, allowing for more trajectories to be
        considered per endpoint rather than just the shortest and longest.

        Returns:
            Number of new trajectories added
        """
        if not self.trajectories:
            return 0

        before_len = len(self.trajectories)

        old_edge_trajectories = self.edge_trajectories
        print(f"Updating from {len(old_edge_trajectories)} trajectories")
        self.edge_trajectories = []  # Clear edge trajectories for this step

        # Group trajectories by their endpoints
        endpoint_to_trajectories = {}
        for traj in old_edge_trajectories:
            if traj.to_delete:
                continue

            traj.discarded = True
            key = traj.get_endpoint_key(self.consider_direction)
            if not key:
                continue

            if key not in endpoint_to_trajectories:
                endpoint_to_trajectories[key] = []
            endpoint_to_trajectories[key].append(traj)

        # Select a broader range of trajectories to expand, not just shortest and longest
        selected_trajectories = self._select_trajectories_to_expand(endpoint_to_trajectories)

        # Mark trajectories not selected as discarded
        for traj in old_edge_trajectories:
            if traj.discarded:
                self.discard_edge_trajectories.append((self.num_step, traj))

        print(f"Total discard: {len(self.discard_edge_trajectories)}")
        print(f"Valid discarded: {len([0 for traj in self.discard_edge_trajectories if not traj[-1].to_delete])}")
        print(f"Selected {len(selected_trajectories)} trajectories for expansion")

        # Process trajectories in parallel
        new_trajectories = self._expand_trajectories_parallel(selected_trajectories)

        print(f"Created {len(new_trajectories)} new trajectories through parallel expansion")

        # Add new trajectories to the tree
        for new_traj in new_trajectories:
            self.trajectories.append(new_traj)
            self._register_trajectory_in_index(new_traj)
            self.edge_trajectories.append(new_traj)

        self.num_step += 1

        # Return count of new trajectories
        return len(self.trajectories) - before_len

    def _select_trajectories_to_expand(self, endpoint_to_trajectories: Dict[Any, List[Trajectory]]) -> List[Trajectory]:
        """
        Select trajectories to expand based on a more inclusive strategy.

        Instead of just selecting shortest and longest trajectories per endpoint,
        this method selects multiple trajectories distributed across the range.

        Args:
            endpoint_to_trajectories: Dictionary mapping endpoints to trajectories

        Returns:
            List of trajectories to expand
        """
        selected_trajectories = []

        for key, trajectories in endpoint_to_trajectories.items():
            # Sort by trajectory length (shorter first)
            trajectories.sort(key=lambda t: len(t.route))

            # Select N trajectories evenly distributed across the range
            n = min(self.max_trajectories_per_endpoint, len(trajectories))
            if n > 0:
                if n == 1:
                    indices = [0]  # Just the shortest
                elif n == 2:
                    indices = [0, len(trajectories) - 1]  # Shortest and longest
                else:
                    # Evenly distribute selections
                    indices = [int(i * (len(trajectories) - 1) / (n - 1)) for i in range(n)]

                for i in indices:
                    trajectories[i].discarded = False
                    selected_trajectories.append(trajectories[i])

        return selected_trajectories

    def _expand_trajectories_parallel(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        Expand multiple trajectories in parallel using a thread pool.

        Args:
            trajectories: List of trajectories to expand

        Returns:
            List of new trajectories created by expansion
        """
        all_new_trajectories = []

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all expansion tasks
            future_to_trajectory = {
                executor.submit(self._safe_expand_trajectory, traj): traj
                for traj in trajectories
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_trajectory):
                try:
                    new_trajectories = future.result()
                    if new_trajectories:
                        all_new_trajectories.extend(new_trajectories)
                except Exception as e:
                    print(f"Error collecting expansion results: {e}")

        return all_new_trajectories

    def _safe_expand_trajectory(self, trajectory: Trajectory) -> List[Trajectory]:
        """
        Safely expand a trajectory, catching any exceptions.

        Args:
            trajectory: The trajectory to expand

        Returns:
            List of new trajectories or empty list if expansion fails
        """
        try:
            return trajectory.get_new_trajectories(max_backtrack=self.max_backtrack)
        except Exception as e:
            print(f"Error expanding trajectory: {e}")
            return []
