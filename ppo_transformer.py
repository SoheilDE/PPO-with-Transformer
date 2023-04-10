import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class PPOTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_heads, num_layers):
        super(PPOTransformer, self).__init__()
        
        self.actor_encoder = nn.Linear(state_dim, hidden_dim)
        self.actor_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads), num_layers=num_layers)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Linear(hidden_dim, action_dim)
        
        self.critic_encoder = nn.Linear(state_dim, hidden_dim)
        self.critic_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads), num_layers=num_layers)
        self.critic_value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        actor_x = F.relu(self.actor_encoder(state))
        actor_x = self.actor_transformer(actor_x.unsqueeze(1)).squeeze(1)
        mean = self.actor_mean(actor_x)
        std = F.softplus(self.actor_std(actor_x))
        actor_dist = Normal(mean, std)
        
        critic_x = F.relu(self.critic_encoder(state))
        critic_x = self.critic_transformer(critic_x.unsqueeze(1)).squeeze(1)
        critic_value = self.critic_value(critic_x)
        
        return actor_dist, critic_value