import struct
from enum import Enum
from math import floor

import numpy as np

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


class PlaceAction:
    def __init__(self, field: list, stone: Stone):
        self.field: list = field
        self.stone: Stone = stone

    def getActionInt(self):
        real_action = self.stone.value - 1
        idx = (self.field[0], self.field[1], real_action)
        return np.ravel_multi_index(idx, (5, 5, 123))


class MoveAction:
    def __init__(self, field, direction: Direction, split):
        self.field: list = field  # x, y
        self.direction: Direction = direction
        self.stone_number = sum(split)
        self.split: list = split  # How to leave down the stones.

    def getActionInt(self):
        real_action = ((inv_splits_dict[tuple(self.split)] + self.direction.value * 30) + 3)
        idx = (self.field[0], self.field[1], real_action)
        return np.ravel_multi_index(idx, (5, 5, 123))


# For now we test only with 5x5 board.
class Tak:
    """
        Use 1 for player1 and -1 for player2.

        See othello/OthelloGame.py for an example implementation.
        """

    possible_splits = 30
    actions_per_field = possible_splits * 4 + 3

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
    def getAction(self, action):
        # action index in 5*5*123 array
        assert action < self.getActionSize()
        action = np.unravel_index(action, (5, 5, 123))
        row = action[0]
        column = action[1]
        real_action = action[2]
        if real_action < 3:
            return PlaceAction([row, column], real_action + 1)

        real_action -= 3
        direction: Direction = Direction(real_action // self.possible_splits)
        move_split = real_action % self.possible_splits

        return MoveAction([row, column], direction, Tak.getSplit(move_split))

    def __init__(self):

        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """

        return np.zeros((5, 5, 43), dtype=int)  # 43 height 21 black, 21 white max 1 capstone on top.

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return 5, 5

    # Action space is 5*5*151

    @staticmethod
    def getActionSize():
        """
        Returns:
            actionSize: number of all possible actions
        """
        return (5 * 5) * ((1 + 2 + 4 + 8 + 15) * 4 + 3)  # 3 place actions, other are moving actions in 4 directions.

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        #assert self.getValidMoves(board, player)[action] != 0
        #Tak.printBoard(board)
        #act = Tak().getAction(action)
        #if type(act) is PlaceAction:
           # print(f"Place stone at row={act.field[0]} col={act.field[1]}")
        #else:
        #    print(
        #        f"Move stone from row={act.field[0]} col={act.field[1]} in direction {act.direction} with split {act.split}")
        board = np.copy(board)
        action = self.getAction(action)
        if type(action) is PlaceAction:
            board[action.field[0]][action.field[1]][0] = player * action.stone
        if type(action) is MoveAction:
            pile = board[action.field[0]][action.field[1]]
            moved_stack = pile[Tak.getPileHeight(pile) - action.stone_number:Tak.getPileHeight(pile)]
            # removing stones from start stack by setting to 0
            movement = 0
            stones = 0
            for amount in action.split:
                start = stones
                stones += amount
                movement += 1
                nextPile = None
                if action.direction == Direction.NORTH:
                    nextPile = board[action.field[0] - movement][action.field[1]]

                elif action.direction == Direction.EAST:
                    nextPile = board[action.field[0]][action.field[1] + movement]

                elif action.direction == Direction.SOUTH:
                    nextPile = board[action.field[0] + movement][action.field[1]]

                elif action.direction == Direction.WEST:
                    nextPile = board[action.field[0]][action.field[1] - movement]

                if movement == len(action.split) and amount == 1 and \
                        moved_stack[-1] == player * Stone.CAPSTONE.value and \
                        (nextPile[Tak.getPileHeight(nextPile)] == Stone.WALL.value or
                         nextPile[Tak.getPileHeight(nextPile)] == -Stone.WALL.value):
                    nextPile[Tak.getPileHeight(nextPile)] = nextPile[Tak.getPileHeight(
                        nextPile)] / Stone.WALL.value  # results in 1 or -1

                # Now we move the stones
                nextPile[Tak.getPileHeight(nextPile):Tak.getPileHeight(nextPile) + amount] = moved_stack[start:stones]
            pile[Tak.getPileHeight(pile) - action.stone_number:Tak.getPileHeight(pile)] = 0

        #Tak.printBoard(board)
        return board, -1 * player

    def getMaxDisplacementInDirection(self, field: list, direction: Direction, board, player) -> int:
        max_displacement = 0
        board_size = 5
        while True:
            movement = max_displacement + 1
            next_stone = None
            next_pile = None
            if direction == Direction.NORTH:
                if field[0] - movement >= 0:
                    next_pile = board[field[0] - movement][field[1]]

            elif direction == Direction.EAST:
                if field[1] + movement < board_size:
                    next_pile = board[field[0]][field[1] + movement]

            elif direction == Direction.SOUTH:
                if field[0] + movement < board_size:
                    next_pile = board[field[0] + movement][field[1]]

            elif direction == Direction.WEST:
                if field[1] - movement >= 0:
                    next_pile = board[field[0]][field[1] - movement]

            if next_pile is not None:
                next_stone = abs(next_pile[Tak.getPileHeight(next_pile) - 1])

            max_displacement += 1
            if next_stone is None or next_stone == Stone.CAPSTONE or next_stone == Stone.WALL:
                break

        return max_displacement - 1

    def getMaxDisplacementInDirectionUsingCapstone(self, field: list, direction: Direction, board, player) -> int:
        max_displacement = 0
        board_size = 5
        while True:
            movement = max_displacement + 1
            next_stone = None
            if direction == Direction.NORTH:
                if field[0] - movement >= 0:
                    next_stone = abs(board[field[0] - movement][field[1]])

            elif direction == Direction.EAST:
                if field[1] + movement < board_size:
                    next_stone = abs(board[field[0]][field[1] + movement])

            elif direction == Direction.SOUTH:
                if field[0] + movement < board_size:
                    next_stone = abs(board[field[0] + movement][field[1]])

            elif direction == Direction.WEST:
                if field[1] - movement >= 0:
                    next_stone = abs(board[field[0]][field[1] - movement])

            if next_stone is None or next_stone == Stone.CAPSTONE:
                break
            if next_stone == Stone.WALL:
                return max_displacement + 1  # because then we flatten the wall.
            max_displacement += 1

        return max_displacement

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        moves = [0] * Tak.getActionSize()

        board_size = 5

        for row in range(0, len(board)):
            for column in range(0, len(board[row])):
                # First checking place actions
                pile = board[row, column]
                if Tak.getPileHeight(board[row, column]) == 0:
                    # moves[PlaceAction([row, column], Stone.WALL).getActionInt()] = 1
                    # moves[PlaceAction([row, column], Stone.CAPSTONE).getActionInt()] = 1
                    moves[PlaceAction([row, column], Stone.FLAT).getActionInt()] = 1

                # Now checking if we can move the pile.
                elif player * pile[Tak.getPileHeight(pile) - 1] > 0:  # pile owned
                    max_stones = min(board_size, Tak.getPileHeight(pile))
                    for direction in Direction:
                        if player * pile[Tak.getPileHeight(pile) - 1] == Stone.CAPSTONE.value:
                            max_displacement = self.getMaxDisplacementInDirectionUsingCapstone([row, column], direction,
                                                                                               board, player)
                            for split in splits_dict.values():
                                if len(split) == max_displacement and sum(split) <= max_stones and split[-1] == 1:
                                    moves[MoveAction([row, column], direction, split).getActionInt()] = 1

                        max_displacement = self.getMaxDisplacementInDirection([row, column], direction, board, player)

                        for split in splits_dict.values():
                            if len(split) <= max_displacement and sum(split) <= max_stones:
                                moves[MoveAction([row, column], direction, split).getActionInt()] = 1
        return moves

    @staticmethod
    def getPileHeight(pile: list):
        height = 0
        while height < 43:
            if pile[height] == 0:
                break
            height += 1
        return height

    @staticmethod
    def printBoard(board):
        for row in range(0, len(board)):
            for column in range(0, len(board[row])):
                # First checking place actions
                pile = board[row, column]
                print("| ", end="")
                for i in range(0, Tak.getPileHeight(pile)):
                    print(f"{pile[i]}", end="")
                print("|", end="")
            print("")
        print("")

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """

        for row in range(0, len(board)):
            for column in range(0, len(board[row])):
                # First checking place actions
                pile = board[row, column]
                if Tak.getPileHeight(pile) == 3:
                    if player * pile[Tak.getPileHeight(pile) - 1] > 0:
                        return 1
                    else:
                        return -1
        return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return player * board  # inverting colors.

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pi_board = np.reshape(pi, (5, 5, 123))
        boards = []

        for i in range(1, 5):  # Rotations
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:  # Reflections
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                boards += [(newB, list(newPi.ravel()))]

        return boards

    @staticmethod
    def stringRepresentation(board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        board_bytes = bytearray()
        for row in range(0, len(board)):
            for column in range(0, len(board[row])):
                # First checking place actions
                pile = board[row, column]
                board_bytes.append("|".encode("ascii")[0])
                # Not using getPileHeight for efficiency
                i = 0
                while True:
                    if pile[i] == 0:
                        break
                    else:
                        board_bytes.append(int(pile[i]).to_bytes(byteorder="big", length=1, signed=True)[0])
                    i += 1
            board_bytes.append("\n".encode("ascii")[0])
        return bytes(board_bytes)

    @staticmethod
    def boardRepresentation(bytes_board: bytes) -> np.ndarray:
        new_board = np.zeros((5, 5, 43), dtype=int)
        row = -1
        column = 0
        ht = 0

        for bt in bytes_board:
            if bt == "|".encode("ascii")[0]:
                row += 1
                ht = 0
            elif bt == "\n".encode("ascii")[0]:
                row = -1
                column += 1
            else:
                new_board[row][column][ht] = bt - 256 if bt > 127 else bt
                ht += 1
        return new_board

    @staticmethod
    def getSplit(move_split):
        return splits_dict[move_split]
