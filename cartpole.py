import gym


class CartPoleEnv:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)