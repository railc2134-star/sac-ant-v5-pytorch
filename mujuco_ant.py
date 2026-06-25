import gymnasium as gym
import numpy as np
import torch
import random
import torch.nn as nn
from collections import deque
import numpy as np
env = gym.make("Ant-v5", render_mode="human")
print(env.observation_space.shape)
class critic1(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(113, 128)
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.input(x))
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

class critic2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(113, 128)
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.input(x))
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

class actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(105, 128)
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 16)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        mean = x[..., :8]
        std = x[..., 8:]
        std = torch.clamp(std, -20, 2)
        std = torch.exp(std)
        return std, mean

class buffer:
    def __init__(self, capcity):
        self.buffer = deque(maxlen=capcity)

    def addd(self, current_obs, action, next_obs, reword, done):
        self.buffer.append((current_obs, action, next_obs, reword, done))

    def randoom(self, batch):
        return random.sample(self.buffer, batch)

    def __len__(self):
        return len(self.buffer)

batch = 64
buf = buffer(10000)
act = actor()
cri1 = critic1()
gamma = 0.99
alpha = 0.2
tar1 = critic1()
tau = 0.005
tar1.load_state_dict(cri1.state_dict())
cri2 = critic2()
tar2 = critic2()
tar2.load_state_dict(cri2.state_dict())
loss = nn.MSELoss()
reward_history = []
optimize1 = torch.optim.Adam(cri1.parameters(), lr=3e-4)
optimize2 = torch.optim.Adam(cri2.parameters(), lr=3e-4)
optimize3 = torch.optim.Adam(act.parameters(), lr=3e-4)
train = False
if train == True:
    for episode in range(10000):
        done = False
        steps = 0
        obs, _ = env.reset()
        total_reword = 0
        while done == False and steps < 1000:
            steps += 1
            std, mean = act(torch.FloatTensor(obs))
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.sample()
            action = torch.tanh(raw_action)
            next_obs, reword, ter, trunc, _ = env.step(action.detach().numpy())
            done = ter or trunc
            buf.addd(obs, action, next_obs, reword, done)
            obs = next_obs
            if len(buf) >= batch:
                buff = buf.randoom(batch)
                current_obs, actions, next_obss, rewords, dones = zip(*buff)
                current_obs_B = torch.FloatTensor(current_obs)
                next_obss_B = torch.FloatTensor(next_obss)
                actions_B = torch.stack(actions)
                rewords_B = torch.FloatTensor(rewords)
                dones_B = torch.LongTensor(dones)
                with torch.no_grad():
                    new_std, new_mean = act(next_obss_B)
                    new_dist = torch.distributions.Normal(new_mean, new_std)
                    new_raw_action = new_dist.sample()
                    new_action = torch.tanh(new_raw_action)
                    per_joint_log_prob = new_dist.log_prob(new_raw_action) - torch.log(1 - torch.tanh(new_raw_action).pow(2) + 1e-6)
                    log_prob = per_joint_log_prob.sum(dim=-1)
                    target = rewords_B + gamma * (torch.min(tar1(next_obss_B, new_action).squeeze(), tar2(next_obss_B, new_action).squeeze()) - alpha * log_prob) * (1 - dones_B)
                optimize1.zero_grad()
                losss1 = loss(cri1(current_obs_B, actions_B).squeeze(), target)
                losss1.backward()
                optimize1.step()
                optimize2.zero_grad()
                losss2 = loss(cri2(current_obs_B, actions_B).squeeze(), target)
                losss2.backward()
                optimize2.step()
                for tw, cw in zip(tar1.parameters(), cri1.parameters()):
                    tw.data.copy_(tw.data * (1 - tau) + cw.data * tau)
                for tw, cw in zip(tar2.parameters(), cri2.parameters()):
                    tw.data.copy_(tw.data * (1 - tau) + cw.data * tau)
                new_std, new_mean = act(current_obs_B)
                new_dist = torch.distributions.Normal(new_mean, new_std)
                new_raw_action = new_dist.rsample()
                new_action = torch.tanh(new_raw_action)
                per_joint_log_prob = new_dist.log_prob(new_raw_action) - torch.log(1 - torch.tanh(new_raw_action).pow(2) + 1e-6)
                log_prob = per_joint_log_prob.sum(dim=-1)
                optimize3.zero_grad()
                losss3 = (alpha * log_prob - torch.min(cri1(current_obs_B, new_action).squeeze(), cri2(current_obs_B, new_action).squeeze())).mean()
                losss3.backward()
                optimize3.step()
            total_reword += reword
        reward_history.append(total_reword)
        print(f"episode = {episode} || steps = {steps} || std = {std.mean().item()} || total_reward = {total_reword.mean()}")
        if episode % 100 == 0:
            print(f"AVG last 100 = {sum(reward_history[-100:]) / 100:.2f}")
        if episode % 50 == 0:
            torch.save(act.state_dict(), 'antacttMUJOCO.pth')
            torch.save(cri1.state_dict(), 'antcri1MUJOCO.pth')
            torch.save(cri2.state_dict(), 'antcri2MUJOCO.pth')
else:
    act.load_state_dict(torch.load('antacttMUJOCO.pth', map_location='cpu'))
    act.eval()
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            with torch.no_grad():
                steps += 1
                std, mean = act(torch.FloatTensor(obs))
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample()
                action = torch.tanh(raw_action)
            obs, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated
            total_reward += reward
        print(f"episode {episode} → reward = {total_reward:.2f},  → steps = {steps}")
