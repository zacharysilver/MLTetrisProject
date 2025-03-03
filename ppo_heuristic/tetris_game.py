# tetris_game.py

import sys
import time
import pygame
from stable_baselines3 import PPO

from tetris_env import TetrisEnv
from heuristic_agent import HeuristicTetrisAgent

def run_rl_agent_mode(model_path="ppo_tetris_colored.zip"):
    env = TetrisEnv()
    model = PPO.load(model_path)
    obs = env.reset()
    done = False
    
    while True:
        env.render()
        if done:
            print("Game Over! (Agent Mode)")
            time.sleep(2)
            break

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    env.close()

def run_user_mode():
    env = TetrisEnv()
    obs = env.reset()
    done = False

    while True:
        env.render()
        if done:
            print("Game Over! (User Mode)")
            time.sleep(2)
            break

        action = get_user_input()
        if action is None:
            action = 0  # default to "no-op"
        obs, reward, done, info = env.step(action)

    env.close()

def run_assisted_mode(model_path="ppo_tetris_colored.zip"):
    env = TetrisEnv()
    model = PPO.load(model_path)
    obs = env.reset()
    done = False

    while True:
        env.render()
        if done:
            print("Game Over! (Assisted Mode)")
            time.sleep(2)
            break

        # Agent's suggestion
        suggested_action, _ = model.predict(obs, deterministic=True)
        print(f"Agent suggests action: {suggested_action}")
        
        user_action = get_user_input()
        if user_action is None:
            action = suggested_action
        else:
            action = user_action

        obs, reward, done, info = env.step(action)

    env.close()

def run_heuristic_mode():
    env = TetrisEnv()
    agent = HeuristicTetrisAgent()
    obs = env.reset()
    done = False

    while True:
        env.render()
        if done:
            print("Game Over! (Heuristic Mode)")
            time.sleep(2)
            break

        # Let the heuristic pick the next action
        action = agent.choose_action(env)
        obs, reward, done, info = env.step(action)

    env.close()

def get_user_input():
    """
    Maps user keyboard inputs to environment actions.
      0 = no-op
      1 = move left
      2 = move right
      3 = rotate
      4 = drop
    Returns None if no relevant key is pressed.
    """
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 1
            elif event.key == pygame.K_RIGHT:
                action = 2
            elif event.key == pygame.K_UP:
                action = 3
            elif event.key == pygame.K_DOWN:
                action = 4
            elif event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
    return action

if __name__ == "__main__":
    """
    Usage:
      python tetris_game.py agent
      python tetris_game.py user
      python tetris_game.py assisted
      python tetris_game.py heuristic
    """
    if len(sys.argv) < 2:
        mode = "agent"
    else:
        mode = sys.argv[1]

    if mode == "agent":
        run_rl_agent_mode()
    elif mode == "user":
        run_user_mode()
    elif mode == "assisted":
        run_assisted_mode()
    elif mode == "heuristic":
        run_heuristic_mode()
    else:
        print("Invalid mode. Use 'agent', 'user', 'assisted', or 'heuristic'.")
