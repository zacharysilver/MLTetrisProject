# ppo_agent.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from tetris_env import TetrisEnv

def train_ppo_agent(total_timesteps=50000, save_path="ppo_tetris_colored"):
    """
    Train a PPO model on the improved TetrisEnv.
    """
    # Create a vectorized environment (just 1 env, but you could do more)
    env = make_vec_env(TetrisEnv, n_envs=1)
    
    # Optionally use a bigger network
    policy_kwargs = dict(net_arch=[256, 256])
    
    # Instantiate PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        policy_kwargs=policy_kwargs
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    model.save(save_path)
    env.close()
    print(f"Model saved to {save_path}.zip")

if __name__ == "__main__":
    # Adjust timesteps as desired
    train_ppo_agent(total_timesteps=50000, save_path="ppo_tetris_colored")
