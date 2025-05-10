from enum import IntEnum

class Tile(IntEnum):
    NO_VISION = 0
    EMPTY = 1
    RECON = 2
    MISSION = 3

class Player(IntEnum):
    SCOUT = 2  # 2**2
    GUARD = 3  # 2**3

class Wall(IntEnum):
    RIGHT = 4  # 2**4 = 16
    BOTTOM = 5  # 2**5 = 32
    LEFT = 6  # 2**6 = 64
    TOP = 7  # 2**7 = 128
