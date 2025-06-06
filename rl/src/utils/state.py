from typing import Optional

import numpy as np
from numpy.typing import NDArray

import torch
from tensordict.tensordict import TensorDict


def unpack_bits(arr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    channel definition:
        0: top wall
        1: left wall
        2: bottom wall
        3: right wall
        4: guard
        5: scout
        6: no vision
        7: is empty
        8: recon point
        9: mission point

    Args:
        arr (NDArray[np.uint8]): array of shape [height, width]
    Output:
        (NDArray[np.uint8]): array of shape [10, height, width]
    """
    unpacked = np.unpackbits(np.expand_dims(arr, 0), axis=0)
    tile_content = np.eye(4, dtype=np.uint8)[unpacked[-2] * 2 + unpacked[-1]].transpose((-1, 0, 1)) # one hot encode last 2 bits
    arr = np.concatenate((unpacked[:-2], tile_content)) # replace last 2 bits with their meanings

    return arr

def prepare_viewcone(
    viewcone: NDArray[np.uint8],
    direction: int,
    remove_self_agent: bool = True,
) -> NDArray[np.uint8]:
    rectified_viewcone = np.zeros((9, 9), dtype=np.uint8)
    rectified_viewcone[2:, 2:7] = viewcone

    rectified_viewcone = unpack_bits(rectified_viewcone)

    if remove_self_agent:
        rectified_viewcone[4][4, 4] = 0 # guard
        rectified_viewcone[5][4, 4] = 0 # scout

    for turn in range(direction):
        # left_bit, bottom_bit, right_bit, top_bit = top_bit, left_bit, bottom_bit, right_bit
        _top = rectified_viewcone[0]
        _left = rectified_viewcone[1]
        _bottom = rectified_viewcone[2]
        _right = rectified_viewcone[3]
        (
            rectified_viewcone[1],
            rectified_viewcone[2],
            rectified_viewcone[3],
            rectified_viewcone[0]
        ) = (
            _top,
            _left,
            _bottom,
            _right
        )

    rectified_viewcone = np.rot90(rectified_viewcone, k=direction, axes=(1, 2))

    return rectified_viewcone

def prepare_map(
    map: NDArray[np.uint8]
) -> NDArray[np.uint8]:

    map = unpack_bits(map)

    return map

class StateManager:
    def __init__(
        self,
        n_frames: Optional[int] = None,
        use_mapped_viewcone: bool = False,
    ):
        self.n_frames = n_frames
        self.use_mapped_viewcone = use_mapped_viewcone

        self.observations: list[dict] = []
        self.maps: list[NDArray[np.uint8]] = []

    def update(self, observation: dict[str, int | NDArray[np.uint8]], map: NDArray[np.uint8] | torch.Tensor):
        """
        Args:
            observation: from env
            map (NDArray[np.uint8] | torch.Tensor): either the reconstructed or the ground truth map.
        """
        self.observations.append(observation)
        self.maps.append(map) # type: ignore

    def __getitem__(self, idx: int) -> TensorDict:
        if self.use_mapped_viewcone:
            if self.n_frames is None:
                observation = self.observations[idx]
                map = self.maps[idx]

                return TensorDict({
                    "map": map,
                    "location": torch.from_numpy(observation['location']),
                    "direction": torch.tensor(observation['direction']),
                    "step": torch.tensor(observation['step'] / 100)
                })
            else:
                maps = torch.zeros((self.n_frames, 12, 31, 31))
                locations = torch.zeros((self.n_frames, 2))
                directions = torch.zeros((self.n_frames, 1))
                steps = torch.zeros((self.n_frames, 1))

                n_available = min(len(self.maps), self.n_frames)
                maps[:n_available] = torch.stack(self.maps[-n_available:]) # type: ignore

                recent_observations = self.observations[-n_available:]

                for i, observation in enumerate(recent_observations):
                    locations[i] = torch.from_numpy(observation['location'])
                    directions[i] = observation['direction']
                    steps[i] = observation['step']

                return TensorDict({
                    "map": maps,
                    "location": locations,
                    "direction": directions,
                    "step": steps,
                    "seq_len": n_available
                })

        else:
            observation = self.observations[idx]

            if self.n_frames is None:
                # Single frame case
                viewcone = prepare_viewcone(
                    observation['viewcone'],
                    observation['direction']
                )
            else:
                # Pre-allocate array for all viewcones
                viewcone = np.zeros((10, self.n_frames, 9, 9), dtype=np.uint8)

                # Vectorized processing of viewcones
                for i, obs in enumerate(self.observations[:-self.n_frames-1:-1]):
                    viewcone[:, -i-1] = prepare_viewcone(
                        obs['viewcone'],
                        obs['direction']
                    )

            map_array = prepare_map(self.maps[idx])

            return TensorDict({
                "viewcone": torch.from_numpy(viewcone),
                "map": torch.from_numpy(map_array),
                "location": torch.from_numpy(observation['location']),
                "direction": torch.tensor(observation['direction']),
                "step": torch.tensor(observation['step'] / 100)
            })
