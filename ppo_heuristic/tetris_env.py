# tetris_env.py

import gym
import pygame
import numpy as np
from gym import spaces

from tetris_gamestate import (TetrisGameState, BOARD_HEIGHT, BOARD_WIDTH, SHAPE_TO_ID)

# For fancy colors, define them again or reuse a dict
SHAPE_COLORS = {
    0: (0,0,0),
    1: (0,255,255),  # I
    2: (255,255,0),  # O
    3: (128,0,128),  # T
    4: (0,255,0),    # S
    5: (255,0,0),    # Z
    6: (0,0,255),    # J
    7: (255,140,0),  # L
}

BLOCK_SIZE = 30

class TetrisEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(TetrisEnv, self).__init__()

        self.game_state = TetrisGameState()
        self.screen = None
        self.clock = None
        self.font = None

        self.action_space = spaces.Discrete(5)  # 0..4
        # observation space: e.g. flatten 0..7
        self.observation_space = spaces.Box(
            low=0, high=7,
            shape=(BOARD_HEIGHT * BOARD_WIDTH,),
            dtype=np.int32
        )

    def reset(self):
        self.game_state.reset()
        return self._get_obs()

    def step(self, action):
        reward, done = self.game_state.step(action)
        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (BOARD_WIDTH*BLOCK_SIZE, BOARD_HEIGHT*BLOCK_SIZE)
            )
            pygame.display.set_caption("Tetris - Option A")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)
        
        # Fill background
        self.screen.fill((0, 0, 0))

        # Draw board
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                val = self.game_state.board[r, c]
                if val != 0:
                    color = SHAPE_COLORS[val]
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (c*BLOCK_SIZE, r*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
                        0
                    )

        # Draw current piece (if not game over)
        if not self.game_state.game_over and self.game_state.current_piece:
            shape_id = SHAPE_TO_ID[self.game_state.current_piece.shape]
            color = SHAPE_COLORS[shape_id]
            for (rr, cc) in self.game_state.current_piece.cells:
                if 0 <= rr < BOARD_HEIGHT and 0 <= cc < BOARD_WIDTH:
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (cc*BLOCK_SIZE, rr*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
                        0
                    )

        # Score / Level
        score_text = f"Score: {self.game_state.score}  Level: {self.game_state.level}"
        text_surface = self.font.render(score_text, True, (255,255,255))
        self.screen.blit(text_surface, (5,5))

        # Game over text
        if self.game_state.game_over:
            go_text = self.font.render("GAME OVER!", True, (255,0,0))
            rect = go_text.get_rect(
                center=(BOARD_WIDTH*BLOCK_SIZE//2, BOARD_HEIGHT*BLOCK_SIZE//2)
            )
            self.screen.blit(go_text, rect)

        pygame.display.flip()
        # Use a speed based on level
        speed = 5 + self.game_state.level * 2
        self.clock.tick(speed)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # --------------------------------
    # Helper
    # --------------------------------
    def _get_obs(self):
        """Flatten the board for the RL agent."""
        return self.game_state.board.flatten()
