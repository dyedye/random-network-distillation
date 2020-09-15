from typing import Any, Sequence
from utils import RunningMeanStd
import gym
from agent import RNDAgent
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta


class Runner:
    def __init__(self, env: Any, agent: Any, save_interval: int = 1000, train_episode: int = 10**9, num_eval_episode: int = 3, episode_len: int = 3000,
                 pre_step: int = 10000, gamma: float = 0.995, int_gamma: float = 0.995, lam: float = 0.97, device=torch.device('cpu'), int_coef: float = 1, ext_coef: float = 0.3,
                 eval_interval: int = 10**4, seed: int = 0):
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        # prepare envs
        self.env = env
        self.env.seed(seed)
        self.env_test = deepcopy(env)
        self.env_test.seed(2**31 - seed)
        self.agent = agent

        # pepare steps
        self.global_step = 0
        self.step_in_episode = 0
        self.episode_so_far = 0

        self.episode_len = episode_len  # length of an episode
        self.num_eval_episode = num_eval_episode
        self.train_episode = train_episode
        self.pre_step = pre_step  # number of steps used to measure variance of states
        self.reward_rms = RunningMeanStd()
        obs_sampled = self.env.reset()
        self.obs_rms = RunningMeanStd(shape=[1] + list(obs_sampled.shape))
        self.device = device
        self.lam = lam
        self.gamma = gamma
        self.int_gamma = int_gamma  # gamma for intrinsic reward
        # ratio of intrinsic and extrinsic rewards
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.reward_in_episode = 0.0
        self.returns = {'step': [], 'return': []}

    def run_episode(self):
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy = \
            [], [], [], [], [], [], [], [], [], []
        self.step_in_episode = 0
        self.reward_in_episode = 0
        obs = self.env.reset()
        done = False

        for _ in range(self.episode_len):
            action, policy, value_ext, value_int = self.agent.get_action(
                obs)
            obs_next, reward, done, info = env.step(2*action)
            self.reward_in_episode += reward
            self.global_step += 1
            self.step_in_episode += 1
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
            obs = obs_next

        _, _, value_ext, value_int = agent.get_action(obs)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)

        total_state = np.stack(total_state)  # (num_episode, state_shape)
        total_action = np.stack(total_action)  # (num_episode)
        total_done = np.stack(total_done)  # (num_episode, )
        total_next_obs = np.stack(total_next_obs)  # (num_episode, state_shape)
        total_int_reward = np.stack(total_int_reward)

        # normalize intrinsic reward
        mean, std, count = np.mean(total_reward), np.std(
            total_reward), len(total_reward)
        self.reward_rms.update_from_moments(mean, std ** 2, count)
        total_int_reward /= self.reward_rms.var

        ext_target, ext_adv = self.gae(reward=total_reward,
                                       done=total_done,
                                       value=total_ext_values,
                                       gamma=self.gamma,
                                       num_step=self.episode_len)
        int_target, int_adv = self.gae(reward=total_int_reward,
                                       done=[0] * self.episode_len,
                                       value=total_int_values,
                                       gamma=self.int_gamma,
                                       num_step=self.episode_len)
        total_adv = int_adv * self.int_coef + ext_adv * self.ext_coef
        self.obs_rms.update(total_next_obs)
        agent.train_model(states=np.float32(total_state),
                          target_ext=ext_target,
                          target_int=int_target,
                          actions=total_action,
                          advs=total_adv,
                          next_states=((total_next_obs - self.obs_rms.mean) /
                                       np.sqrt(self.obs_rms.var)).clip(-5, 5),
                          log_pi_old=total_policy,  # TODO: fix this
                          num_step=self.episode_len)

    def evaluate(self, steps):
        """ 複数エピソード環境を動かし，平均収益を記録する． """

        returns = []
        for _ in range(self.num_eval_episode):
            state = self.env_test.reset()
            done = False
            episode_return = 0.0
            step = 0
            while (not done):
                step += 1
                action = self.agent.exploit(state)
                state, reward, done, _ = self.env_test.step(2*action)
                episode_return += reward

            returns.append(episode_return)

        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)

        print(f'Num steps: {steps:<6}   '
              f'Num episode: {self.episode_so_far}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    def plot(self):
        """ 平均収益のグラフを描画する． """
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Return', fontsize=24)
        plt.tick_params(labelsize=18)
        plt.title(f'{self.env.unwrapped.spec.id}', fontsize=24)
        plt.tight_layout()
        plt.savefig('figure.png')

    def start(self):
        self.start_time = time()
        self.prepare_normalization_coeff()
        print('Start Training')
        for episode in range(self.train_episode):
            self.episode_so_far = episode
            self.run_episode()
            if episode % self.eval_interval:
                self.evaluate(steps=self.global_step)
            if episode % (self.eval_interval * 10):
                self.plot()
        print('Finished')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))

    def prepare_normalization_coeff(self):
        states = []
        for _ in range(self.pre_step):
            action = self.env.action_space.sample()
            state, reward, done, info = self.env.step(action)
            states.append(state)
        states = np.array(states)
        self.obs_rms.update(states)

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
                     rnd_output_size=(10, ), state_shape=obs_shape, action_shape=action_shape)
    runner = Runner(env=env, agent=agent)
    runner.start()
