import gymnasium
from gymnasium import envs
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN
from cnn import CNN

from datetime import datetime, time, timedelta
import time
import argparse
import itertools
import snake_env
import os

DATE_FORMAT = "%m-%d %H:%M%S"

RUNS_DIR = "runs"

os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Agent:

    def __init__(self, hyperparameter_set, use_cnn=False):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.use_cnn = use_cnn

        self.env_id = hyperparameters["env_id"]
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        self.stop_on_reward = hyperparameters["stop_on_reward"]
        self.nodes = hyperparameters["nodes"]
        self.env_make_params = hyperparameters.get("env_make_params", {})
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None

        # Path to Run info
        suffix = "_cnn" if self.use_cnn else ""
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}{suffix}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}{suffix}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}{suffix}.png')

    def train(self, render=False):

        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        envs = gymnasium.vector.SyncVectorEnv(
            [lambda: gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params, cnn=self.use_cnn, size=15),
             lambda: gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params, cnn=self.use_cnn, size=15),
             lambda: gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params, cnn=self.use_cnn, size=15),
             lambda: gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params, cnn=self.use_cnn, size=15)]
        )

        observation_shape = envs.observation_space.shape
        num_envs = envs.num_envs
        num_actions = envs.single_action_space.n
        rewards_per_episode = []
        scores_per_episode = []

        if self.use_cnn:
            policy_dqn = CNN(observation_shape, 15, num_actions, self.nodes).to(device)
        else:
            
            num_states = envs.single_observation_space.shape[0]
            policy_dqn = DQN(num_states, num_actions, self.nodes).to(device)

        memory = ReplayMemory(self.replay_memory_size)

        epsilon = self.epsilon_init

        if self.use_cnn:
            target_dqn = CNN(observation_shape, 15, num_actions, self.nodes).to(device)
        else:
            target_dqn = DQN(num_states, num_actions, self.nodes).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        epsilon_history = []

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        best_reward = -9999999

        step_count = 0
        completed_episodes = 0
        episode_rewards = np.zeros(num_envs, dtype=np.float32)
        states, _ = envs.reset()

        for vector_step in itertools.count():
            states_tensor = torch.tensor(
                states,
                dtype=torch.float,
                device=device,
            )

            random_mask = np.random.random(num_envs) < epsilon

            if random_mask.all():
                actions = envs.action_space.sample()
            else:
                with torch.no_grad():
                    greedy_actions = (
                        policy_dqn(states_tensor)
                        .argmax(dim=1)
                        .cpu()
                        .numpy()
                    )

                actions = envs.action_space.sample()
                actions[~random_mask] = greedy_actions[~random_mask]

            new_states, rewards, terminations, truncations, infos = envs.step(actions)
            dones = terminations | truncations
            new_states_tensor = torch.tensor(
                new_states,
                dtype=torch.float,
                device=device,
            )
            rewards_tensor = torch.tensor(
                rewards,
                dtype=torch.float,
                device=device,
            )
            episode_rewards += rewards

            for i in range(num_envs):
                action_tensor = torch.tensor(
                    actions[i],
                    dtype=torch.int64,
                    device=device,
                )
                memory.append((
                    states_tensor[i],
                    action_tensor,
                    new_states_tensor[i],
                    rewards_tensor[i],
                    bool(dones[i]),
                ))

                if dones[i]:
                    episode_reward = float(episode_rewards[i])
                    score_values = infos.get("score")
                    score = (
                        float(score_values[i])
                        if score_values is not None else 0.0
                    )
                    rewards_per_episode.append(episode_reward)
                    scores_per_episode.append(score)
                    completed_episodes += 1
                    episode_rewards[i] = 0.0

                    if episode_reward > best_reward:
                        log_message = (
                            f"{datetime.now().strftime(DATE_FORMAT)}: "
                            f"New best reward {episode_reward:0.1f} at episode "
                            f"{completed_episodes} with score {score:0.1f}, "
                            "saving model..."
                        )
                        print(log_message)
                        with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_reward = episode_reward
                    elif completed_episodes % 500 == 0:
                        log_message = (
                            f"{datetime.now().strftime(DATE_FORMAT)}: Reward "
                            f"{episode_reward:0.1f} at episode "
                            f"{completed_episodes} with score {score:0.1f}, "
                            "saving model..."
                        )
                        print(log_message)
                        with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

            epsilon = max(
                epsilon * self.epsilon_decay,
                self.epsilon_min,
            )
            epsilon_history.append(epsilon)

            if len(memory) >= self.mini_batch_size and vector_step % 10 == 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                if step_count >= self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

            
            states = new_states
            step_count += num_envs

            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=20):
                self.save_graph(
                    rewards_per_episode,
                    epsilon_history,
                    scores_per_episode,
                )
                last_graph_update_time = current_time

    def test(self, render=False):
        env = gymnasium.make(self.env_id, render_mode="human" if render else "", **self.env_make_params, cnn=self.use_cnn, size=15)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        if self.use_cnn:
            policy_dqn = CNN(env.observation_space.shape, 15, num_actions).to(device)
        else:
            policy_dqn = DQN(num_states, num_actions, self.nodes).to(device)

        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):

                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, truncated, info = env.step(int(action.item()))
                score = info.get("score", 0)

                # Accumulate reward
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)

                # move to new state
                state = new_state

            log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Reward {episode_reward:0.1f} at episode {episode} with score {score:0.1f}"
            print(log_message)
            with open(self.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')

    def save_graph(self, rewards_per_episode, epsilon_history, scores_per_episode):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(131) # plot on a 1 row x 3 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(132) # plot on a 1 row x 3 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplot(133) # plot on a 1 row x 3 col grid, at cell 3
        plt.ylabel('MeanScores')
        mean_scores = np.zeros(len(scores_per_episode))
        for x in range(len(mean_scores)):
            mean_scores[x] = np.mean(scores_per_episode[max(0, x-99):(x+1)])
        plt.subplot(133) # plot on a 1 row x 3 col grid,
        plt.plot(mean_scores)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack Tensors to create batch tensors
        # tensor([1, 2, 3])
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
                        
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]


        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or test model. ")
    parser.add_argument('hyperparameters', help=' ')
    parser.add_argument("--train", help="Training mode", action="store_true")
    parser.add_argument("--cnn", help="Use CNN", action="store_true")
    args = parser.parse_args()

    if args.cnn:
        dql = Agent(hyperparameter_set=args.hyperparameters, use_cnn=True)
    else:
        dql = Agent(hyperparameter_set=args.hyperparameters, use_cnn=False)

    if args.train:
        dql.train(render=False)
    else:
        dql.test(render=True)
