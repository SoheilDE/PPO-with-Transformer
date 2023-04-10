import torch

from ppo_transformer import PPOTransformer
from cartpole import CartPoleEnv


def evaluate_ppo_transformer(env, model, num_episodes):
    total_rewards = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                actor_dist, _ = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = actor_dist.mean
            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            total_reward += reward
        total_rewards.append(total_reward)
    avg_reward = sum(total_rewards) / num_episodes
    return avg_reward


if __name__ == '__main__':
    env = CartPoleEnv()
    model = PPOTransformer(state_dim=env.state_dim, action_dim=env.action_dim, 
                           hidden_dim=64, num_heads=2, num_layers=2)
    model.load_state_dict(torch.load('ppo_transformer.pt'))
    
    avg_reward = evaluate_ppo_transformer(env, model, num_episodes=10)
    print(f'Average reward over 10 episodes: {avg_reward:.2f}')