import os
import numpy as np
import torch
from src.Agents.mappo import MAPPO
from src.Utils.logger import logger
from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3
from datetime import datetime


class MAPPOTrainer:
    def __init__(self, config, scenario='simple_adversary_v3'):
        self.env_name = scenario
        self.env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=False)
        self.env.reset()

        self.config = config
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)
        logger.info(f"Initialized environment '{self.env_name}' with {self.n_agents}")

        self.state_dims = {agent: self.env.observation_space(agent).shape[0] for agent in self.agents}
        self.action_dims = {agent: self.env.action_space(agent).n for agent in self.agents}

        self.save_dir = os.path.join("models", scenario)
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"Models will be saved to {self.save_dir}")

        self.log_dir = os.path.join("results", scenario)
        os.makedirs(self.log_dir, exist_ok=True)
        logger.info(f"Logs will be saved to {self.log_dir}")

        self.max_episodes = config['n_episodes']
        self.max_timesteps = config['max_training_timesteps']
        self.print_freq = config['print_freq']
        self.save_freq = config['save_model_freq']

        self.mappo = MAPPO(self.state_dims, self.n_agents, self.action_dims, self.env, config, scenario)

        self.best_score = float('-inf')
        self.rewards_log = []

    def train(self):
        start_time = datetime.now().replace(microsecond=0)
        logger.info(f"Training started at {start_time}")

        total_steps = 0
        i_episode = 0

        while total_steps <= self.config['max_training_timesteps']:
            obs, _ = self.env.reset()
            episode_rewards = {agent: 0.0 for agent in self.agents}

            for step in range(1, self.config['max_ep_len'] + 1):
                actions, logprobs, values = self.mappo.select_action(obs)
                next_obs, rewards, terminations, truncations, _ = self.env.step(actions)

                for agent in self.agents:
                    self.mappo.store_transition(
                        agent_name=agent,
                        state=torch.tensor(obs[agent], dtype=torch.float32),
                        global_state=self.mappo.aggregate_obs(obs),
                        action=actions[agent],
                        logprob=logprobs[agent],
                        value=values[agent],
                        reward=rewards[agent],
                        done=terminations[agent] or truncations[agent]
                    )
                    episode_rewards[agent] += rewards[agent]

                obs = next_obs
                total_steps += 1

                if all(terminations[a] or truncations[a] for a in self.agents):
                    break

            avg_reward = sum(episode_rewards.values()) / self.n_agents
            self.rewards_log.append(avg_reward)

            if total_steps % self.config['update_timestep'] == 0:
                self.mappo.update()

            if total_steps % self.print_freq == 0:
                logger.info(f"Episode {i_episode}, Timestep {total_steps}, Avg Reward: {avg_reward:.2f}")

            if total_steps % self.save_freq == 0:
                if avg_reward > self.best_score:
                    self.best_score = avg_reward
                    logger.info(f"New best avg reward {avg_reward:.2f} at timestep {total_steps}. Saving model.")
                    self.mappo.save(self.save_dir)
            
            i_episode += 1

        self.env.close()
        np.save(os.path.join(self.log_dir, f"{self.env_name}_rewards.npy"), np.array(self.rewards_log))
        logger.info(f"Training complete. Rewards saved to {self.log_dir}")
