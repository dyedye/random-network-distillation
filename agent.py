from rnd import RNDModel
from ppo import PPO
import torch


class RNDAgent:
    def __init__(self, rnd_input_size, rnd_output_size, state_shape, action_shape, device=torch.device('cpu')):
        self.rnd = RNDModel(input_size=rnd_input_size,
                            output_size=rnd_output_size)
        self.ppo = PPO(state_shape=state_shape,
                       action_shape=action_shape, device=device)
        self.device = device

    def calc_intrinsic_reward(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        feature_target = self.rnd.target(obs)
        feature = self.rnd.predictor(obs)
        reward = (feature_target - feature).pow(2).sum(1) / 2
        # TODO: check whether this detach is correct or not
        return reward.detach().cpu().numpy()

    def get_action(self, obs):
        """
        Returns (action, policy, value_ext, value_int)
        """
        obs = torch.FloatTensor(obs).to(self.device)
        value_ext, value_int = self.ppo.critic(obs)
        action, policy = self.ppo.explore(obs)
        return action, policy, value_ext.data.cpu().numpy(), value_int.data.cpu().numpy()
