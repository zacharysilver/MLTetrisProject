# tetris_gamestate.py

import numpy as np
import random

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

TETROMINOS = {
    'I': [
        [(0,0), (0,1), (0,2), (0,3)],
        [(0,1), (1,1), (2,1), (3,1)],
    ],
    'O': [
        [(0,0), (0,1), (1,0), (1,1)],
    ],
    'T': [
        [(0,0), (0,1), (0,2), (1,1)],
        [(0,1), (1,0), (1,1), (2,1)],
        [(1,0), (1,1), (1,2), (0,1)],
        [(0,0), (1,0), (2,0), (1,1)],
    ],
    'S': [
        [(0,1), (0,2), (1,0), (1,1)],
        [(0,0), (1,0), (1,1), (2,1)],
    ],
    'Z': [
        [(0,0), (0,1), (1,1), (1,2)],
        [(0,1), (1,0), (1,1), (2,0)],
    ],
    'J': [
        [(0,0), (1,0), (1,1), (1,2)],
        [(0,1), (0,2), (1,1), (2,1)],
        [(1,0), (1,1), (1,2), (2,2)],
        [(0,1), (1,1), (2,1), (2,0)],
    ],
    'L': [
        [(0,2), (1,0), (1,1), (1,2)],
        [(0,1), (1,1), (2,1), (2,2)],
        [(1,0), (1,1), (1,2), (2,0)],
        [(0,0), (0,1), (1,1), (2,1)],
    ],
}

SHAPE_TO_ID = {
    'I': 1,
    'O': 2,
    'T': 3,
    'S': 4,
    'Z': 5,
    'J': 6,
    'L': 7
}

class Tetromino:
    def __init__(self, shape, row=0, col=BOARD_WIDTH//2 - 1, rotation=0):
        self.shape = shape
        self.row = row
        self.col = col
        self.rotation = rotation
    
    @property
    def cells(self):
        rots = TETROMINOS[self.shape]
        rot_pat = rots[self.rotation % len(rots)]
        return [(r + self.row, c + self.col) for (r, c) in rot_pat]

class TetrisGameState:
    """
    Pure logic-based Tetris state, no rendering or PyGame code.
    """
    def __init__(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int32)
        self.current_piece = None
        self.game_over = False
        self.score = 0
        self.lines_cleared_total = 0
        
        # For "level" logic if you want
        self.level = 1
        self.lines_for_next_level = 10

    def clone(self):
        """
        Return a deep copy of just the game logic state.
        This is safe for BFS or heuristic enumeration.
        """
        new_state = TetrisGameState()
        new_state.board = self.board.copy()
        if self.current_piece:
            # manual piece copy
            new_state.current_piece = Tetromino(
                self.current_piece.shape,
                self.current_piece.row,
                self.current_piece.col,
                self.current_piece.rotation
            )
        new_state.game_over = self.game_over
        new_state.score = self.score
        new_state.lines_cleared_total = self.lines_cleared_total
        new_state.level = self.level
        new_state.lines_for_next_level = self.lines_for_next_level
        return new_state

    def reset(self):
        self.board[:] = 0
        self.game_over = False
        self.score = 0
        self.lines_cleared_total = 0
        self.level = 1
        self.current_piece = self._get_new_piece()

    def step(self, action):
        """
        action in {0..4}:
          0: no-op
          1: move left
          2: move right
          3: rotate
          4: drop
        Returns (reward, done) so we can pass it up to a Gym env if needed.
        """
        reward = 0.0

        # If game_over, do nothing
        if self.game_over:
            return (0.0, True)

        # Manual actions
        if action == 1:  # left
            self._move_piece(0, -1)
        elif action == 2:  # right
            self._move_piece(0, 1)
        elif action == 3:  # rotate
            self._rotate_piece()
        elif action == 4:  # drop
            while not self._would_collision(1, 0):
                self._move_piece(1, 0)
            self._lock_piece()
            cleared = self._clear_lines()
            self.score += cleared * 100
            self.lines_cleared_total += cleared
            reward += cleared
            self._check_level_up()
            self.current_piece = self._get_new_piece()
            if self.current_piece is None or self._would_collision(0,0, piece=self.current_piece):
                self.game_over = True
                reward -= 5.0
            return (reward, self.game_over)

        # Auto-fall
        if not self._move_piece(1, 0):
            # lock
            self._lock_piece()
            cleared = self._clear_lines()
            self.score += cleared * 100
            self.lines_cleared_total += cleared
            reward += cleared
            self._check_level_up()
            self.current_piece = self._get_new_piece()
            if self.current_piece is None or self._would_collision(0,0, piece=self.current_piece):
                self.game_over = True
                reward -= 5.0

        return (reward, self.game_over)

    # --------------------------------------------------------
    # Internal logic
    # --------------------------------------------------------
    def _check_level_up(self):
        if self.lines_cleared_total >= self.level * self.lines_for_next_level:
            self.level += 1

    def _get_new_piece(self):
        for _ in range(30):
            shape = random.choice(list(TETROMINOS.keys()))
            piece = Tetromino(shape, row=0, col=BOARD_WIDTH//2 - 2, rotation=0)
            if not self._would_collision(0, 0, piece=piece):
                return piece
        self.game_over= True

    def _would_collision(self, dx, dy, rotation=None, piece=None):
        if piece is None:
            piece = self.current_piece
        old_rot = piece.rotation
        new_rot = old_rot if rotation is None else rotation
        rots = TETROMINOS[piece.shape]
        rot_pat = rots[new_rot % len(rots)]
        for (r_off, c_off) in rot_pat:
            r = piece.row + r_off + dx
            c = piece.col + c_off + dy
            if c < 0 or c >= BOARD_WIDTH or r >= BOARD_HEIGHT:
                return True
            if r >= 0 and self.board[r, c] != 0:
                return True
        return False

    def _move_piece(self, dx, dy):
        if self._would_collision(dx, dy):
            return False
        self.current_piece.row += dx
        self.current_piece.col += dy
        return True

    def _rotate_piece(self):
        old = self.current_piece.rotation
        new = old + 1
        if not self._would_collision(0, 0, rotation=new):
            self.current_piece.rotation = new

    def _lock_piece(self):
        shape_id = SHAPE_TO_ID[self.current_piece.shape]
        for (r, c) in self.current_piece.cells:
            if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                self.board[r, c] = shape_id

    def _clear_lines(self):
        lines_cleared = 0
        for r in range(BOARD_HEIGHT):
            if all(self.board[r, c] != 0 for c in range(BOARD_WIDTH)):
                lines_cleared += 1
                # shift above rows down
                self.board[1:r+1, :] = self.board[0:r, :]
                self.board[0, :] = 0
        return lines_cleared
