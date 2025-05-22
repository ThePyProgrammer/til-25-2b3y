import random
from typing import Optional

from .timeout import timeout, TimeoutError


@timeout(20)
def _reset_environment(env, seed, **kwargs):
    return env.reset(seed=seed, **kwargs)

def reset_environment(env, seed: Optional[int] = None, **kwargs):
    if seed is None:
        seed = random.randint(0, 999999)

    while True:
        try:
            _reset_environment(env, seed, **kwargs)
            break
        except TimeoutError:
            seed += 1

    return env
