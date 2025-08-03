import os
import numpy as np
import torch
from src.Agents.mappo import MAPPO
from src.Utils.logger import logger
from pettingzoo.atari import space_invaders_v2
from pettingzoo.mpe import simple_tag_v3
from datetime import datetime


class MAPPOTrainer:
    def __init__(self, config, scenario=None):
        self.env_name = scenario
        self.env = space_invaders_v2.parallel_env()

        self.config = config
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)
        logger.info(f"Initialized environment '{self.env_name}' with {self.n_agents}")

        observations, infos = self.env.reset()
        self.actor_dims = {agent: observations[agent].shape[0] if len(observations[agent].shape)==1 else np.prod(observations[agent].shape) for agent in self.env.agents}
        self.n_actions = {agent: self.env.action_space(agent).n for agent in self.env.agents}

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

        self.mappo = MAPPO(self.n_actions, self.env, config, scenario)
        logger.info(f"MAPPO initialized with {self.n_agents} agents in scenario '{scenario}'")

        self.best_score = float('-inf')
        self.rewards_log = {agent: [] for agent in self.agents}


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
                    proc = self.mappo.preprocess(obs[agent])  # shape: [1, 1, 84, 84]
                    state_to_store = proc.squeeze(0)
                    self.mappo.store_transition(
                        agent_name=agent,
                        state=torch.tensor(state_to_store.cpu(), dtype=torch.float32),
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

            for agent in self.agents:
                self.rewards_log[agent].append(episode_rewards[agent])


            if total_steps % self.config['update_timestep'] == 0:
                logger.info(f"Training at episode {i_episode}, total steps {total_steps}, rewards: {episode_rewards}")
                self.mappo.update()

            #if total_steps % self.print_freq == 0:
            #    logger.info(f"Episode {i_episode}, Timestep {total_steps}, Avg Reward: {avg_reward:.2f}")

            if total_steps % self.save_freq == 0:
                self.mappo.save(self.save_dir)
            
            i_episode += 1

        
        np.save(os.path.join(self.log_dir, f"{self.env_name}_rewards.npy"), np.array(self.rewards_log))
        logger.info(f"Training complete. Rewards saved to {self.log_dir}")
        self.env.close()