# heuristic_agent.py

import numpy as np
# We only import TetrisGameState, TETROMINOS, etc. from tetris_gamestate,
# NOT from tetris_env (which has PyGame stuff that can't be pickled).
from tetris_gamestate import (
    TetrisGameState,
    BOARD_WIDTH,
    BOARD_HEIGHT,
    TETROMINOS,
    SHAPE_TO_ID
)

# Define the action constants that match TetrisGameState.step(...)
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE = 3
ACTION_DROP = 4

class HeuristicTetrisAgent:
    """
    A BFS-like agent that tries all possible placements for the current piece,
    using only the *game state* (env.game_state.clone()). This avoids
    copying PyGame surfaces in the environment itself.
    """
    def __init__(self):
        self._action_plan = []

    def choose_action(self, env):
        """
        Called each step. If we already have a plan of actions, pop the next one.
        Otherwise, compute the best move sequence for the current piece.
        env can be either a tetris_env or tetris_gamestate.
        """
        # If we have an action plan left over, use it:
        if self._action_plan:
            return self._action_plan.pop(0)

        # Compute a new action plan:
        best_sequence = self._find_best_move_sequence(env)
        self._action_plan = best_sequence

        # If no sequence found, default to no-op:
        if not self._action_plan:
            return ACTION_NOOP

        return self._action_plan.pop(0)

    def _find_best_move_sequence(self, env):
        """
        1-step lookahead:
        - For each rotation (0..max),
        - For each column (0..BOARD_WIDTH-1),
        - Build an action sequence: rotate piece, move horizontally, then drop.
        - Simulate on a *clone* of the game state to see the result (score, board).
        - Pick the best sequence by a heuristic.
        """
        best_score = -999999
        best_sequence = []

        game_state = env if isinstance(env, TetrisGameState) else env.game_state # The pure logic from TetrisEnv
        shape = game_state.current_piece.shape
        max_rotations = len(TETROMINOS[shape])

        for rotation_idx in range(max_rotations):
            for target_col in range(BOARD_WIDTH):
                # Build an action sequence for (rotation_idx, target_col)
                seq = self._build_action_sequence(game_state, rotation_idx, target_col)

                # Simulate
                sim_state, total_reward, done = self._simulate_actions(game_state, seq)

                # Evaluate the final board after those actions
                heuristic_score = self._evaluate_state(sim_state)
                # Combine immediate reward + final heuristic
                total_score = total_reward + heuristic_score

                if total_score > best_score:
                    best_score = total_score
                    best_sequence = seq

        return best_sequence

    def _build_action_sequence(self, game_state, desired_rotation, target_col):
        """
        Construct a list of actions that:
         - Rotates current piece from its current rotation to desired_rotation,
         - Moves horizontally until piece.col == target_col (if possible),
         - Then calls DROP (action=4).

        Return a list like [3, 3, 1, 1, 4] for rotate, rotate, move-left, move-left, drop.
        """
        seq = []

        # 1) Figure out how many rotations needed
        current_rot = game_state.current_piece.rotation
        needed_rotations = (desired_rotation - current_rot) % 4
        for _ in range(needed_rotations):
            seq.append(ACTION_ROTATE)

        # 2) Figure out horizontal moves
        current_col = game_state.current_piece.col
        col_diff = target_col - current_col
        if col_diff > 0:
            # Need to move right col_diff times
            seq.extend([ACTION_RIGHT] * col_diff)
        elif col_diff < 0:
            # Move left abs(col_diff) times
            seq.extend([ACTION_LEFT] * abs(col_diff))

        # 3) Finally, drop
        seq.append(ACTION_DROP)

        return seq

    def _simulate_actions(self, game_state, action_sequence):
        """
        Clone the TetrisGameState, apply the actions in order, and return:
         - the final sim_state,
         - the total reward gained,
         - whether game ended (done).
        """
        sim_state = game_state.clone()
        total_reward = 0.0
        done = False

        for act in action_sequence:
            reward, done = sim_state.step(act)
            total_reward += reward
            if done:
                break

        return sim_state, total_reward, done

    def _evaluate_state(self, sim_state):
        """
        Heuristic that checks the final board: holes, height, etc.
        Return a numeric value (higher = better).
        """
        board = sim_state.board
        # Simple example: a weighted sum of lines_cleared_total minus holes, etc.
        lines_cleared = sim_state.lines_cleared_total
        holes = self._count_holes(board)
        aggregate_height = self._compute_aggregate_height(board)
        bumpiness = self._compute_bumpiness(board)

        # Adjust weights as you like
        score = (lines_cleared * 10) - (holes * 5) - (aggregate_height * 0.5) - (bumpiness * 0.5)
        return score

    def _count_holes(self, board):
        """A hole is an empty cell with at least one filled cell above it in the same column."""
        holes = 0
        rows, cols = board.shape
        for c in range(cols):
            found_filled = False
            for r in range(rows):
                if board[r, c] != 0:
                    found_filled = True
                elif board[r, c] == 0 and found_filled:
                    holes += 1
        return holes

    def _compute_aggregate_height(self, board):
        rows, cols = board.shape
        total_height = 0
        for c in range(cols):
            col_height = 0
            for r in range(rows):
                if board[r, c] != 0:
                    col_height = rows - r
                    break
            total_height += col_height
        return total_height

    def _compute_bumpiness(self, board):
        rows, cols = board.shape
        heights = []
        for c in range(cols):
            h = 0
            for r in range(rows):
                if board[r, c] != 0:
                    h = rows - r
                    break
            heights.append(h)
        bumpiness = 0
        for i in range(len(heights)-1):
            bumpiness += abs(heights[i] - heights[i+1])
        return bumpiness
