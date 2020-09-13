from typing import Any, Sequence
from utils import RunningMeanStd
import gym
from agent import RNDAgent
from logger import DummyLogger
import numpy as np
import torch


class Runner:
    def __init__(self, env: Any, agent: Any, save_interval: int = 1000, max_episode: int = 10 ^ 7, logger=DummyLogger, episode_len: int = 100,
                 pre_step: int = 100, gamma: float = 0.99, int_gamma: float = 0.99, lam: float = 0.99, device=torch.device('cpu')):
        self.save_interval = save_interval
        self.env = env
        self.agent = agent
        self.logger = logger
        self.global_step = 0
        self.step_in_episode = 0
        self.episode_len = episode_len
        self.max_episode = max_episode
        self.pre_step = pre_step
        self.reward_rms = RunningMeanStd()
        obs_sampled = self.env.reset()
        self.obs_rms = RunningMeanStd(shape=[1] + list(obs_sampled.shape))
        self.device = device
        self.lam = lam
        self.int_gamma = int_gamma  # gamma for intrinsic reward

    def run_episode(self):
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy = \
            [], [], [], [], [], [], [], [], [], []
        self.step_in_episode = 0
        obs = self.env.reset()
        done = False

        for _ in range(self.episode_len):
            action, policy, value_ext, value_int = self.agent.get_action(
                obs)  # TODO: fix this
            obs_next, reward, done, info = env.step(action)
            self.global_step += 1
            self.step_in_episode += 1
            self.logger.log(
                reward=reward, global_step=self.global_step, step_in_episode=self.step_in_episode)

            int_reward = agent.calc_intrinsic_reward(
                (obs_next - self.obs_rms.mean) / np.sqrt(self.obs_rms.var).clip(-5, 5))

            total_next_obs.append(obs_next)
            total_int_reward.append(int_reward)
            total_state.append(obs)
            total_reward.append(reward)
            total_done.append(done)
            total_action.append(action)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)

        _, _, value_ext, value_int = agent.get_action(
            np.float32(obs) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)

        total_state = np.stack(total_state)  # (num_episode, state_shape)
        total_action = np.stack(total_action)  # (num_episode)
        total_done = np.stack(total_done)  # (num_episode, )
        total_next_obs = np.stack(total_next_obs)  # (num_episode, state_shape)
        total_logging_policy = np.array(total_policy)

        ext_target, ext_adv = self.gae(reward=total_reward,
                                       done=total_done,
                                       value=total_int_values,
                                       gamma=self.int_gamma,
                                       num_step=self.num_step)

    def start(self):
        self.prepare_normalization_coeff()
        for _ in range(self.max_episode):
            self.run_episode()

    def prepare_normalization_coeff(self):
        for _ in range(self.pre_step):
            action = self.env.action_space.sample()
            state, reward, done, info = self.env.step(action)
            self.obs_rms.update(state)

    def gae(self, reward: Sequence, done: Sequence, value: Sequence, gamma: float, num_step: int):
        """Returns (discounted_return, advantage)"""
        adv_tmp = 0
        discounted_return = [None] * num_step
        for t in range(num_step - 1, -1, -1):
            delta = reward[t] + gamma * value[t + 1] * (1 - done[t]) - value[t]
            adv_tmp = delta + gamma * self.lam * (1 - done[t]) * adv_tmp
            discounted_return[t] = adv_tmp + value[t]
        discounted_return = np.array(discounted_return, dtype='float32')
        adv = discounted_return - np.array(value[:-1], dtype='float32')
        return discounted_return, adv


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    obs_sample = env.reset()
    obs_shape = obs_sample.shape
    action_shape = env.action_space.shape
    agent = RNDAgent(device=torch.device('cpu'), rnd_input_size=obs_shape,
                     rnd_output_size=action_shape, state_shape=obs_shape, action_shape=action_shape)
    runner = Runner(env=env, agent=agent)
    runner.start()
