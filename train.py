import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
from stable_baselines3 import PPO
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import time
import os
import torch  # Added this import

class PyBoyEnv(gym.Env):
    def __init__(self, rom_path="SuperMarioLand.gb"):
        super().__init__()
        self.pyboy = PyBoy(rom_path, window="null")
        self.pyboy.set_emulation_speed(0)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_map = ["A", "B", "LEFT", "RIGHT", "UP", "DOWN"]
        self.prev_screen_hash = None

    def reset(self, seed=None, options=None):
        self.pyboy.button("START")
        self.pyboy.tick()
        self.pyboy.button_release("START")
        for _ in range(60):
            self.pyboy.tick()
        self.prev_screen_hash = None
        return self._get_obs(), {}

    def step(self, action):
        button = self.action_map[action]
        self.pyboy.button(button)
        self.pyboy.tick()
        self.pyboy.button_release(button)
        for _ in range(15):
            self.pyboy.tick()
        obs = self._get_obs()
        reward = self._get_reward(obs)
        done = self._is_done(obs)
        return obs, reward, done, False, {}

    def _get_obs(self):
        screen = self.pyboy.screen.ndarray
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[..., np.newaxis]

    def _get_reward(self, obs):
        reward = 0.5
        current_screen_hash = hash(obs.tobytes())
        if self.prev_screen_hash and current_screen_hash != self.prev_screen_hash:
            reward += 1.0
        else:
            reward -= 0.1
        self.prev_screen_hash = current_screen_hash
        return reward

    def _is_done(self, obs):
        if self.prev_screen_hash is not None:
            return self.prev_screen_hash == hash(obs.tobytes())
        return False

    def close(self):
        self.pyboy.stop(save=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Train or play Mario RL agent with PyBoy")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA (GPU) if available")
    parser.add_argument("--model_path", type=str, default="mario_ppo_model.zip", help="Path to save/load the model")
    parser.add_argument("--play", action="store_true", help="Play using the loaded model instead of training")
    return parser.parse_args()

def play_model(env, model):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    print("Starting play mode...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        time.sleep(0.05)
    print(f"Play ended. Total reward: {total_reward}")
    env.close()

def train_model(env, model, total_timesteps, model_path):
    start_time = time.time()
    with tqdm(total=total_timesteps, desc="Training", unit="timestep") as pbar:
        def callback(_locals, _globals):
            current_timesteps = _locals["self"].num_timesteps
            pbar.n = current_timesteps
            pbar.refresh()
            elapsed_time = time.time() - start_time
            if current_timesteps > 0:
                time_per_timestep = elapsed_time / current_timesteps
                remaining_timesteps = total_timesteps - current_timesteps
                time_left = remaining_timesteps * time_per_timestep
                pbar.set_postfix({"Time Left": f"{time_left:.0f}s"})
            return True
        model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = PyBoyEnv()
    total_timesteps = 10_000

    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model = PPO.load(args.model_path, env=env, device=device)
    else:
        print(f"No model found at {args.model_path}. Starting new training.")
        model = PPO("CnnPolicy", env, verbose=1, device=device)

    if args.play:
        play_model(env, model)
    else:
        train_model(env, model, total_timesteps, args.model_path)