import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn
import numpy as np
from collections import deque
import random
import os
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers.frame_stack import FrameStack  # Corrected import
from gym.wrappers import GrayScaleObservation  # Compatible with gymnasium
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Hyperparameters
STATE_SIZE = (4, 84, 84)
ACTION_SIZE = len(SIMPLE_MOVEMENT)
LEARNING_RATE = 0.00025
GAMMA = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Environment Wrappers
def wrap_env(env):
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ResizeObservation(env, (84, 84))       # gymnasium wrapper
    env = GrayScaleObservation(env)               # gym wrapper, compatible
    env = FrameStack(env, num_stack=4)            # Corrected gymnasium wrapper
    return env

# Agent
class MarioAgent:
    def __init__(self, action_size, use_cuda):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        if use_cuda and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
        print(f"Using device: {self.device}")
        self.action_size = action_size
        self.q_network = QNetwork(action_size).to(self.device)
        self.target_network = QNetwork(action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.step = 0

    def act(self, state, play=False):
        if play or random.random() > self.epsilon:
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.max(1)[1].item()
        return random.randrange(self.action_size)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return None, None

        transitions = self.memory.sample(BATCH_SIZE)
        batch = tuple(zip(*transitions))

        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + (1 - dones) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

        return q_values.mean().item(), loss.item()

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())

def main():
    parser = argparse.ArgumentParser(description="Train or play Super Mario Bros with DDQN")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--resume", action="store_true", help="Resume training from model_path")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    parser.add_argument("--play", action="store_true", help="Play using the model at model_path")
    parser.add_argument("--model_path", type=str, default="mario_model.pth", help="Path to save/load model")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    # Initialize environment and agent
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", render_mode="human" if args.render else "rgb_array")
    env = wrap_env(env)
    agent = MarioAgent(ACTION_SIZE, use_cuda=args.cuda)

    # Load model if resuming or playing
    if (args.resume or args.play) and os.path.exists(args.model_path):
        agent.load(args.model_path)
        print(f"Loaded model from {args.model_path}")

    if args.play:
        # Play mode
        state, _ = env.reset()  # Modern API: (obs, info)
        total_reward = 0
        done = False
        while not done:
            if args.render:
                env.render()
            action = agent.act(state, play=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            done = terminated or truncated
            if args.debug:
                print(f"Action: {action}, Reward: {reward}, Info: {info}")
        print(f"Total Reward: {total_reward}")
    else:
        # Training mode
        for episode in range(args.episodes):
            state, _ = env.reset()  # Modern API: (obs, info)
            total_reward = 0
            done = False
            while not done:
                if args.render:
                    env.render()
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.memory.push(state, action, reward, next_state, done)
                q, loss = agent.learn()
                total_reward += reward
                state = next_state

                if args.debug and q is not None:
                    print(f"Episode: {episode}, Step: {agent.step}, Q: {q:.2f}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.3f}")

            print(f"Episode {episode + 1}/{args.episodes}, Total Reward: {total_reward}")
            if (episode + 1) % 10 == 0:
                agent.save(args.model_path)
                print(f"Saved model to {args.model_path}")

    env.close()

if __name__ == "__main__":
    main()