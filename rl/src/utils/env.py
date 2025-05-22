import random
from typing import Optional

from .timeout import timeout, TimeoutError


@timeout(20)
def _reset_environment(env, seed):
    return env.reset(seed=seed)

def reset_environment(env, seed: Optional[int] = None):
    if seed is None:
        seed = random.randint(0, 999999)

    while True:
        try:
            _reset_environment(env, seed)
            break
        except TimeoutError:
            seed += 1

    return env
