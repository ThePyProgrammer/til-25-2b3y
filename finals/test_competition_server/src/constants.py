from typing import Final

# Task Handler
# total number of data points per task
NUM_DATA_POINTS: Final[int] = 50
# max time per test case for scoring
MAX_TIME_PER_TEST_CASE: Final[float] = 2.0
# weightage of accuracy/reward
PERFORMANCE_WEIGHT: Final[float] = 0.75
SPEED_WEIGHT: Final[float] = 0.25

# Competition Server
# per-round timeout, in seconds
RL_TIME_CUTOFF: Final[float] = 2.0
# number of rounds to run per match
NUM_ROUNDS: Final[int] = 4
# number of items to queue for each special mission
QUEUE_ITEMS_PER_MISSION: Final[int] = 5
