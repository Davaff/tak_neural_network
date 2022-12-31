from enum import Enum

import numpy as np

possible_splits = 30
actions_per_field = possible_splits * 4 + 3

splits_dict = {
    0: [1],
    1: [2],
    2: [1, 1],
    3: [3],
    4: [2, 1],
    5: [1, 2],
    6: [1, 1, 1],
    7: [4],
    8: [3, 1],
    9: [2, 2],
    10: [2, 1, 1],
    11: [1, 3],
    12: [1, 2, 1],
    13: [1, 1, 2],
    14: [1, 1, 1, 1],
    15: [5],
    16: [4, 1],
    17: [3, 2],
    18: [3, 1, 1],
    19: [2, 3],
    20: [2, 2, 1],
    21: [2, 1, 2],
    22: [2, 1, 1, 1],
    23: [1, 4],
    24: [1, 3, 1],
    25: [1, 2, 2],
    26: [1, 2, 1, 1],
    27: [1, 1, 3],
    28: [1, 1, 2, 1],
    29: [1, 1, 1, 2]
}

inv_splits_dict = {tuple(v): k for k, v in splits_dict.items()}

board_glyphs_dict = {
    1: "S",
    2: "W",
    3: "C"
}


class Stone(Enum):  # positive values for white, negative for black
    EMPTY = 0
    FLAT = 1
    WALL = 2
    CAPSTONE = 3


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


# Actions
# 0, 1, 2 = Place FLAT, WALL, CAPSTONE
# For the others make real_action - 3
# Then: real_action / 4 gives direction
# real_action % 30 gives move in the direction with 30 possibilities 0-30:
# 0 -> 1 stone moved
# 1-2 -> 2 stones moved, 1 = both on first field 2 = divided
# 3-6 -> 3 stones moved
# 7-14 -> 4 stones moved
# 30 - -> 5 stones moved

board_size = 5


class PlaceAction:
    stones_dict = {Stone.FLAT: "F", Stone.WALL: "S", Stone.CAPSTONE: "C"}

    def __init__(self, field: list, stone: Stone):
        self.field: list = field
        self.stone: Stone = stone

    def getActionInt(self):
        real_action = self.stone.value - 1
        idx = (self.field[0], self.field[1], real_action)
        return np.ravel_multi_index(idx, (5, 5, 123))

    def getTakticianCommand(self) -> str:
        st = self.stones_dict[self.stone]
        col = chr(97 + self.field[1])
        row = board_size - self.field[0]   # Taktician uses reverse indices.
        return f"{st}{col}{row}"


class MoveAction:

    dirs_dict = {Direction.NORTH: "+", Direction.EAST: ">", Direction.SOUTH: "-", Direction.WEST: "<"}

    def __init__(self, field: list, direction: Direction, split: list):
        self.field: list = field  # x, y
        self.direction: Direction = direction
        self.stone_number: int = sum(split)
        self.split: list = split  # How to leave down the stones.

    def getActionInt(self):
        real_action = ((inv_splits_dict[tuple(self.split)] + self.direction.value * 30) + 3)
        idx = (self.field[0], self.field[1], real_action)
        return np.ravel_multi_index(idx, (5, 5, 123))

    def getTakticianCommand(self) -> str:
        direct = self.dirs_dict[self.direction]
        col = chr(97 + self.field[1])
        row = board_size - self.field[0]  # Taktician uses reverse indices.
        spl = ""
        for sp in self.split:
            spl += f"{sp}"
        return f"{self.stone_number}{col}{row}{direct}{spl}"


class TakState:
    def __init__(self, board: np.ndarray, stones_white: int, stones_black: int,
                 capstones_white: int, capstones_black: int, moves_no_place: int,
                 curr_player: int):
        self.board: np.ndarray = board
        self.stones_white: int = stones_white
        self.stones_black: int = stones_black
        self.capstones_white: int = capstones_white
        self.capstones_black: int = capstones_black
        self.moves_no_place: int = moves_no_place
        self.curr_player: int = curr_player

    def canPlaceStone(self) -> bool:
        if self.curr_player == 1:  # white
            return self.stones_white > 0
        if self.curr_player == -1:  # white
            return self.stones_black > 0
        return False

    def canPlacCapstone(self) -> bool:
        if self.curr_player == 1:  # white
            return self.capstones_white > 0
        if self.curr_player == -1:  # black
            return self.capstones_black > 0
        return False
