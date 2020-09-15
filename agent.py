from rnd import RNDModel
from ppo import PPO
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class RNDAgent:
    def __init__(self, rnd_input_size, rnd_output_size, state_shape, action_shape, device=torch.device('cpu'),
                 epoch: int = 10, batch_size: int = 64, update_proportion=1, ent_coef: float = 0.0, max_grad_norm: float = 0.5,
                 lr: float = 3e-4):
        self.rnd = RNDModel(input_size=rnd_input_size,
                            output_size=rnd_output_size)
        self.ppo = PPO(state_shape=state_shape,
                       action_shape=action_shape, device=device)
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.update_proportion = update_proportion
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.parameters = []
        self.parameters += list(self.ppo.actor.parameters()) + \
            list(self.ppo.critic.parameters())
        self.parameters += list(self.rnd.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=lr)

    def calc_intrinsic_reward(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        feature_target = self.rnd.target(obs)
        feature = self.rnd.predictor(obs)
        reward = (feature_target - feature).pow(2).sum(1) / 2
        return reward.detach().cpu().numpy()

    def get_action(self, obs):
        """
        Returns (action, policy, value_ext, value_int)
        """
        obs = torch.FloatTensor(obs).to(self.device)
        value_int, value_ext = self.ppo.critic(obs)
        action, policy = self.ppo.explore(obs)
        return action, policy, value_ext.data.cpu().numpy(), value_int.data.cpu().numpy()

    def train_model(self, states, target_ext, target_int, actions, advs, next_states, log_pi_old, num_step: int):
        actions = torch.FloatTensor(actions).to(self.device)
        target_ext = torch.FloatTensor(target_ext).to(self.device)
        target_int = torch.FloatTensor(target_int).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advs = torch.FloatTensor(advs).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        states = torch.FloatTensor(states).to(self.device)
        log_pi_old = torch.FloatTensor(states).to(self.device)

        forward_mse = nn.MSELoss(reduction='none')

        sample_indices = np.arange(num_step)
        for i in range(self.epoch):
            np.random.shuffle(sample_indices)
            for j in range(int(num_step / self.batch_size)):
                sample_idx = sample_indices[self.batch_size *
                                            j: self.batch_size * (j + 1)]
                predict_next_state_feature, target_next_state_feature = self.rnd(
                    next_states[sample_idx])
                forward_loss = forward_mse(
                    predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                mask = torch.rand(self.batch_size).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor)
                forward_loss = (forward_loss * mask).sum() / \
                    torch.max(mask.sum(), torch.Tensor([1])).to(self.device)

                value_int, value_ext = self.ppo.critic(states[sample_idx])
                action, policy = self.ppo.explore(states[sample_idx])
                action = torch.FloatTensor(action).to(self.device)
                log_pis = self.ppo.actor.evaluate_log_pi(
                    states=states[sample_idx], actions=action)
                ratio = torch.exp(log_pis - log_pi_old[sample_idx])
                surr1 = -ratio * advs[sample_idx]
                surr2 = -torch.clamp(
                    ratio,
                    1.0 - self.ppo.clip_eps,
                    1.0 + self.ppo.clip_eps
                ) * advs[sample_idx]
                actor_loss = torch.max(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(
                    value_ext, target_ext[sample_idx])
                critic_int_loss = F.mse_loss(
                    value_int, target_int[sample_idx])
                critic_loss = critic_ext_loss + critic_int_loss

                # calculate entropy
                entropy = -log_pis.mean()

                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * \
                    entropy + forward_loss

                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
                self.optimizer.step()

    def exploit(self, state):
        return self.ppo.exploit(state=state)
