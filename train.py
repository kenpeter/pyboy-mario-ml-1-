import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers.frame_stack import FrameStack as GymFrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation as GymGrayScaleObservation
from gymnasium.wrappers.resize_observation import ResizeObservation as GymResizeObservation
from pyboy import PyBoy
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
import torch
import torch.nn as nn
from torchvision import transforms as T

# Frame Skipping Wrapper
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

# Custom Observation Wrapper to handle PyBoy screen data
class PyBoyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)

    def observation(self, observation):
        return observation

# Base PyBoy Environment
class PyBoyEnv(gym.Env):
    def __init__(self, rom_path="SuperMarioLand.gb", render=False):
        super().__init__()
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null")
        self.pyboy.set_emulation_speed(0)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        self.action_map = ["A", "B", "LEFT", "RIGHT", "UP", "DOWN"]
        self.prev_screen_hash = None
        self.render = render

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
        self.pyboy.tick()
        obs = self._get_obs()
        reward = self._get_reward(obs)
        done = self._is_done(obs)
        return obs, reward, done, False, {}

    def _get_obs(self):
        screen = self.pyboy.screen.ndarray
        return screen

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

# MarioNet (Neural Network for Mario Agent)
class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input, model="online"):
        return self.online(input)

# Mario Agent
class Mario:
    def __init__(self, state_dim, action_dim, save_dir, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.device = device

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def save(self):
        save_path = os.path.join(self.save_dir, f"mario_net_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(self.net.state_dict(), save_path)

def apply_wrappers(env):
    env = PyBoyObservationWrapper(env)
    env = SkipFrame(env, skip=4)
    env = GymGrayScaleObservation(env)
    env = GymResizeObservation(env, shape=(84, 84))
    env = GymFrameStack(env, num_stack=4)
    return env

def parse_args():
    parser = argparse.ArgumentParser(description="Train or play Mario RL agent with PyBoy")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA (GPU) if available")
    parser.add_argument("--model_path", type=str, default="models/mario_net.chkpt", help="Path to save/load the model")
    parser.add_argument("--play", action="store_true", help="Play using the loaded model instead of training")
    parser.add_argument("--render", action="store_true", help="Render the game window (SDL2)")
    return parser.parse_args()

def play_model(env, mario):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    print("Starting play mode...")
    while not done:
        action = mario.act(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if env.render:
            time.sleep(0.05)
    print(f"Play ended. Total reward: {total_reward}")
    env.close()

def train_model(env, mario, total_timesteps, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    obs, _ = env.reset()
    start_time = time.time()
    with tqdm(total=total_timesteps, desc="Training", unit="timestep") as pbar:
        for step in range(total_timesteps):
            action = mario.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            obs = next_obs

            pbar.n = step + 1
            pbar.refresh()
            elapsed_time = time.time() - start_time
            if step > 0:
                time_per_timestep = elapsed_time / step
                remaining_timesteps = total_timesteps - step
                time_left = remaining_timesteps * time_per_timestep
                pbar.set_postfix({"Time Left": f"{time_left:.0f}s"})

            if mario.curr_step % mario.save_every == 0:
                mario.save()

            if done:
                obs, _ = env.reset()

    mario.save()
    print(f"Model saved to {os.path.dirname(model_path)}")
    env.close()

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_env = PyBoyEnv(render=args.render)
    env = apply_wrappers(base_env)
    total_timesteps = 90_000

    state_dim = (4, 84, 84)
    action_dim = env.action_space.n
    save_dir = os.path.dirname(args.model_path) or "./models"
    mario = Mario(state_dim, action_dim, save_dir, device)

    if args.play:
        if os.path.exists(args.model_path):
            print(f"Loading model from {args.model_path}")
            mario.net.load_state_dict(torch.load(args.model_path))
            mario.net.eval()
        else:
            print(f"Error: No model found at {args.model_path}")
            exit(1)
        play_model(env, mario)
    else:
        train_model(env, mario, total_timesteps, args.model_path)