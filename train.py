import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader

from ppo_transformer import PPOTransformer
from cartpole import CartPoleEnv


def train_ppo_transformer(env, model, optimizer, epochs, batch_size, clip_ratio, value_coef, entropy_coef, gamma):
    for epoch in range(epochs):
        states, actions, log_probs, rewards, dones, next_states = [], [], [], [], [], []
        state = env.reset()
        while True:
            actor_dist, critic_value = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = actor_dist.sample()
            next_state, reward, done, info = env.step(action.item())
            
            log_prob = actor_dist.log_prob(action).sum(dim=-1)
            entropy = actor_dist.entropy().sum(dim=-1)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            
            state = next_state
            
            if done:
                break
        
        actor_dist, critic_value = model(torch.tensor(next_states[-1], dtype=torch.float32).unsqueeze(0))
        next_value = critic_value.detach().item() if not done else 0.0
        
        returns, advantages = [], []
        G = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            returns.append(G)
            delta = reward + gamma * G - next_value
            advantages.append(delta)
            G = delta + gamma * value_coef * next_value
        
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        advantages = torch.tensor(list(reversed(advantages)), dtype=torch.float32)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        
        dataset = torch.utils.data.TensorDataset(states, actions, log_probs, returns, advantages)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for state, action, old_log_prob, target_return, advantage in dataloader:
            actor_dist, critic_value = model(state)
            
            ratio = torch.exp(actor_dist.log_prob(action).sum(dim=-1) - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(critic_value.squeeze(), target_return)
            
            entropy_loss = -entropy_coef * actor_dist.entropy().mean()
            
            loss = actor_loss + value_coef * critic_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
if __name__ == '__main__':
    env = CartPoleEnv()
    model = PPOTransformer(state_dim=env.state_dim, action_dim=env.action_dim, 
                           hidden_dim=64, num_heads=2, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    train_ppo_transformer(env, model, optimizer, epochs=500, batch_size=64, clip_ratio=0.2, 
                          value_coef=0.5, entropy_coef=0.01, gamma=0.99)