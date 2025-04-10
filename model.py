import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
num_states = 54
num_actions = 4
LR = 0.001
GAMMA = 0.99
LAMBDA = 0.95
EPS = 0.2
NUM_EPOCHS = 10

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(num_states, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

class PPO:
    def __init__(self):
        self.policy_net = PolicyNet().to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = []

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.policy_net(state)
            probs = F.softmax(logits, dim=1)
            action_dist = distributions.Categorical(probs)
            action = action_dist.sample()
            action_prob = probs[0, action.item()].item()
        return action.item(), value.item(), action_prob

    def store_transition(self, state, action, reward, next_state, done, value, action_prob):
        self.memory.append((state, action, reward, next_state, done, value, action_prob))

    def compute_advantages(self, rewards, values, dones):
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                v_next = 0
            else:
                v_next = values[t + 1]
            delta = rewards[t] + GAMMA * v_next * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def learn(self):
        if not self.memory:
            return
        states = [m[0] for m in self.memory]
        actions = [m[1] for m in self.memory]
        rewards = [m[2] for m in self.memory]
        dones = [m[4] for m in self.memory]
        values = [m[5] for m in self.memory]
        old_action_probs = [m[6] for m in self.memory]

        advantages = self.compute_advantages(rewards, values, dones)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        returns = [adv + val for adv, val in zip(advantages, values)]

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        old_action_probs = torch.tensor(old_action_probs, dtype=torch.float).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(device)
        returns = torch.tensor(returns, dtype=torch.float).to(device)

        for _ in range(NUM_EPOCHS):
            logits, values = self.policy_net(states)
            probs = F.softmax(logits, dim=1)
            action_dist = distributions.Categorical(probs)
            new_action_probs = action_dist.probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = new_action_probs / (old_action_probs + 1e-10)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy = action_dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.05 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []