import queue

import numpy as np

from Utility import *


# For now, we test only with 5x5 board.
class Tak:

    @staticmethod
    def getAction(action):
        assert action < Tak.getActionSize()
        action = np.unravel_index(action, (5, 5, 123))

        row = action[0]
        column = action[1]
        real_action = action[2]
        if real_action < 3:
            return PlaceAction([row, column], Stone(real_action + 1))

        real_action -= 3
        direction: Direction = Direction(real_action // possible_splits)
        move_split = real_action % possible_splits

        return MoveAction([row, column], direction, Tak.getSplit(move_split))

    @staticmethod
    def getInitState() -> TakState:
        return TakState(np.zeros((5, 5, 43), dtype=int), 21, 21, 1, 1, 0, 1)

    @staticmethod
    def getBoardSize() -> (int, int):
        return 5, 5

    @staticmethod
    def getActionSize() -> int:
        return (5 * 5) * ((1 + 2 + 4 + 8 + 15) * 4 + 3)  # 3 place actions, other are moving actions in 4 directions.

    @staticmethod
    def getNextState(curr_state: TakState, action: int) -> TakState:

        new_state = TakState(np.copy(curr_state.board), curr_state.stones_white, curr_state.stones_black,
                             curr_state.capstones_white, curr_state.capstones_black, curr_state.moves_no_place,
                             curr_state.curr_player)

        player = curr_state.curr_player
        orig_player = player
        action = Tak.getAction(action)
        if curr_state.stones_white == 21 or curr_state.stones_black == 21:
            player = -player  # first move

        if type(action) is PlaceAction:
            new_state.board[action.field[0]][action.field[1]][0] = player * action.stone.value
            new_state.moves_no_place = 0
            if player == 1:  # white
                if action.stone == Stone.CAPSTONE:
                    new_state.capstones_white -= 1
                else:
                    new_state.stones_white -= 1
            elif player == -1:  # black
                if action.stone == Stone.CAPSTONE:
                    new_state.capstones_black -= 1
                else:
                    new_state.stones_black -= 1

        if type(action) is MoveAction:
            new_state.moves_no_place += 1
            pile = new_state.board[action.field[0]][action.field[1]]
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
                    nextPile = new_state.board[action.field[0] - movement][action.field[1]]

                elif action.direction == Direction.EAST:
                    nextPile = new_state.board[action.field[0]][action.field[1] + movement]

                elif action.direction == Direction.SOUTH:
                    nextPile = new_state.board[action.field[0] + movement][action.field[1]]

                elif action.direction == Direction.WEST:
                    nextPile = new_state.board[action.field[0]][action.field[1] - movement]

                if movement == len(action.split) and amount == 1 and \
                        moved_stack[-1] == player * Stone.CAPSTONE.value and \
                        (nextPile[Tak.getPileHeight(nextPile)] == Stone.WALL.value or
                         nextPile[Tak.getPileHeight(nextPile)] == -Stone.WALL.value):
                    nextPile[Tak.getPileHeight(nextPile)] = nextPile[Tak.getPileHeight(nextPile)] / Stone.WALL.value

                # Now we move the stones
                nextPile[Tak.getPileHeight(nextPile):Tak.getPileHeight(nextPile) + amount] = moved_stack[start:stones]
            pile[Tak.getPileHeight(pile) - action.stone_number:Tak.getPileHeight(pile)] = 0

        new_state.curr_player = orig_player * -1
        return new_state

    @staticmethod
    def getMaxDisplacementInDirection(field: list, direction: Direction, board: np.ndarray) -> int:
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
            if next_stone is None or next_stone == Stone.CAPSTONE.value or next_stone == Stone.WALL.value:
                break

        return max_displacement - 1

    @staticmethod
    def getMaxDisplacementInDirectionUsingCapstone(field: list, direction: Direction, board: np.ndarray) -> int:
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

            if next_stone is None or next_stone == Stone.CAPSTONE.value:
                break
            if next_stone == Stone.WALL.value:
                return max_displacement + 1  # because then we flatten the wall.
            max_displacement += 1

        return max_displacement

    @staticmethod
    def getValidMoves(state: TakState):
        moves = [0] * Tak.getActionSize()

        board_size = 5

        can_place_stone = state.canPlaceStone()
        can_place_capstone = state.canPlacCapstone()
        for row in range(0, len(state.board)):
            for column in range(0, len(state.board[row])):
                # First checking place actions
                pile = state.board[row, column]
                pile_height = Tak.getPileHeight(pile)
                if pile_height == 0:
                    if can_place_stone:
                        moves[PlaceAction([row, column], Stone.FLAT).getActionInt()] = 1
                        if state.stones_white == 21 or state.stones_black == 21:
                            continue  # First move, only flat playing allowed.
                        moves[PlaceAction([row, column], Stone.WALL).getActionInt()] = 1
                    if can_place_capstone:
                        moves[PlaceAction([row, column], Stone.CAPSTONE).getActionInt()] = 1
                # Now checking if we can move the pile.
                elif state.curr_player * pile[pile_height - 1] > 0 and not (state.stones_white == 21 or state.stones_black == 21):  # pile owned
                    max_stones = min(board_size, pile_height)
                    for direction in Direction:
                        if state.curr_player * pile[pile_height - 1] == Stone.CAPSTONE.value:
                            max_displacement = Tak.getMaxDisplacementInDirectionUsingCapstone([row, column], direction,
                                                                                              state.board)

                            for split in splits_dict.values():
                                if len(split) == max_displacement and sum(split) <= max_stones and split[-1] == 1:
                                    moves[MoveAction([row, column], direction, split).getActionInt()] = 1

                        max_displacement = Tak.getMaxDisplacementInDirection([row, column], direction, state.board)

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
        max_height = 0
        for row in range(0, len(board)):
            for column in range(0, len(board[row])):
                height = Tak.getPileHeight(board[row, column])
                if height > max_height:
                    max_height = height

        for row in range(0, len(board)):
            for column in range(0, len(board[row])):
                pile = board[row, column]
                height = Tak.getPileHeight(pile)
                print("|", end="")
                for i in range(0, height):
                    stone = board_glyphs_dict[abs(pile[i])]
                    player = "W" if pile[i] > 0 else "B"
                    print(f"({stone}{player})", end="")
                for i in range(height, max_height):
                    print(f"    ", end="")
            print("|")
        print("")

    @staticmethod
    def getGameEnded(state: TakState):
        """
        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        """
        visited = np.zeros((5, 5), dtype=int)

        for row in range(0, len(state.board)):
            for column in range(0, len(state.board[row])):
                owner, touched_dirs = Tak.checkPath(row, column, visited, state.board)
                if (Direction.NORTH in touched_dirs and Direction.SOUTH in touched_dirs) or \
                        (Direction.WEST in touched_dirs and Direction.EAST in touched_dirs):
                    return state.curr_player * owner

        if state.moves_no_place > 5 * 5 * 20:
            return 0.001
        return 0

    @staticmethod
    def checkPath(row: int, column: int, visited: np.ndarray, board: np.ndarray) -> (int, list):
        touched_dirs = []

        pile = board[row, column]
        pile_height = Tak.getPileHeight(pile)
        if pile_height == 0:
            return 1, touched_dirs
        owner = 1 if pile[pile_height - 1] > 0 else -1

        positions = queue.Queue()
        positions.put([row, column])
        while not positions.empty():
            new_pos = positions.get_nowait()
            if visited[new_pos[0]][new_pos[1]] == 1:
                continue

            pile = board[new_pos[0], new_pos[1]]
            pile_height = Tak.getPileHeight(pile)

            if pile_height == 0:
                continue
            new_owner = 1 if pile[pile_height - 1] > 0 else -1

            if abs(pile[pile_height - 1]) == Stone.WALL.value or new_owner != owner:
                continue

            visited[new_pos[0]][new_pos[1]] = 1

            # Pile makes road and is owned by starting owner
            if new_pos[0] == 0:  # Cannot go up
                touched_dirs.append(Direction.NORTH)
            else:
                positions.put([new_pos[0] - 1, new_pos[1]])

            if new_pos[0] == board_size - 1:  # Cannot go down
                touched_dirs.append(Direction.SOUTH)
            else:
                positions.put([new_pos[0] + 1, new_pos[1]])

            if new_pos[1] == 0:  # Cannot go left
                touched_dirs.append(Direction.WEST)
            else:
                positions.put([new_pos[0], new_pos[1] - 1])

            if new_pos[1] == board_size - 1:  # Cannot go left
                touched_dirs.append(Direction.EAST)
            else:
                positions.put([new_pos[0], new_pos[1] + 1])

        return owner, touched_dirs

    @staticmethod
    def getCanonicalForm(state: TakState) -> TakState:
        new_state = TakState(np.copy(state.board), state.stones_white, state.stones_black,
                             state.capstones_white, state.capstones_black, state.moves_no_place,
                             state.curr_player)
        # We invert the players
        if state.curr_player == 1:
            return new_state
        else:
            new_state.board *= -1
            new_state.curr_player = 1
            # Switching stones.
            tmp_white_stones = new_state.stones_white
            tmp_white_cap = new_state.capstones_white
            new_state.stones_white = new_state.stones_black
            new_state.capstones_white = new_state.capstones_black
            new_state.stones_black = tmp_white_stones
            new_state.capstones_black = tmp_white_cap

        return new_state

    @staticmethod
    def getSymmetries(board, pi):
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
    def stringRepresentation(board: np.ndarray):
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
        column = -1
        row = 0
        ht = 0

        for bt in bytes_board:
            if bt == "|".encode("ascii")[0]:
                column += 1
                ht = 0
            elif bt == "\n".encode("ascii")[0]:
                column = -1
                row += 1
            else:
                new_board[row][column][ht] = bt - 256 if bt > 127 else bt
                ht += 1
        return new_board

    @staticmethod
    def getSplit(move_split: int):
        return splits_dict[move_split]

    @staticmethod
    def parseTakticianResponse(resp: str):
        assert len(resp) >= 2
        i = 0
        stone = None
        stack = 0
        if resp[i] == "F":
            stone = Stone.FLAT
            i += 1
        elif resp[i] == "S":
            stone = Stone.WALL
            i += 1
        elif resp[i] == "C":
            stone = Stone.CAPSTONE
            i += 1
        elif "1" <= resp[i] <= "8":
            stack = int(resp[i])
            i += 1
        else:
            stone = Stone.FLAT

        x = ord(resp[i]) - ord("a")
        i += 1
        y = ord(resp[i]) - ord("1")
        y = board_size - y - 1
        i += 1

        if i == len(resp):
            return PlaceAction([y, x], stone)

        direct = None
        if resp[i] == "+":
            direct = Direction.NORTH
        elif resp[i] == "-":
            direct = Direction.SOUTH
        elif resp[i] == ">":
            direct = Direction.EAST
        elif resp[i] == "<":
            direct = Direction.WEST
        if stack == 0:
            stack = 1

        i += 1
        slides = []
        while i < len(resp):
            slides.append(int(resp[i]))
            i += 1
        if len(slides) == 0:
            slides.append(stack)

        return MoveAction([y, x], direct, slides)
