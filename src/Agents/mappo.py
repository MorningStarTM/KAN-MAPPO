from src.Agents.agent import PPO
from src.Utils.logger import logger
import torch

class MAPPO:
    def __init__(self, actor_dims, n_agents, n_actions, env, config, scenario='default'):
        self.agents = {}
        self.agent_ids = env.possible_agents
        self.n_agents = n_agents
        self.critic_dim = 0

        for i in self.agent_ids:
            self.critic_dim += env.observation_spaces[i].shape[0]
        
        logger.info(f"Total critic dimension: {self.critic_dim}")

        for agent_idx, agent_name in enumerate(self.agent_ids):
            logger.info(f"Initializing PPO for {agent_name}")
            self.agents[agent_name] = PPO(
                state_dim=actor_dims[agent_name],
                all_state_dim=self.critic_dim,
                action_dim=n_actions[agent_name],
                config=config
            )

        logger.info(f"MAPPO initialized with {self.n_agents} agents in scenario '{scenario}'")

    def select_action(self, observations):
        actions = {}
        logprobs = {}
        values = {}
        global_states = self.aggregate_obs(observations)
        for agent_name in self.agent_ids:
            action, logprob, value = self.agents[agent_name].select_action(observations[agent_name], global_states)
            actions[agent_name] = action
            logprobs[agent_name] = logprob
            values[agent_name] = value
        return actions, logprobs, values
    
    def aggregate_obs(self, obs_dict):
        """
        Aggregates observations from all agents into a single global state tensor.

        Args:
            obs_dict (dict): Dictionary of {agent_id: observation (np.array or tensor)}
            device (torch.device or None): Optional device to put the tensor on

        Returns:
            torch.Tensor: Concatenated global state [obs0, obs1, ..., obsN]
        """
        obs_list = [torch.tensor(obs, dtype=torch.float32) for obs in obs_dict.values()]
        global_obs = torch.cat(obs_list, dim=-1)
        return global_obs


    def store_transition(self, agent_name, state, global_state, action, logprob, value, reward, done):
        agent = self.agents[agent_name]
        agent.buffer.states.append(state)
        agent.buffer.global_states.append(global_state)
        agent.buffer.actions.append(torch.tensor(action).to(agent.device))
        agent.buffer.logprobs.append(torch.tensor(logprob).to(agent.device))
        agent.buffer.state_values.append(torch.tensor(value).to(agent.device))
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)

    def update(self):
        for agent_name in self.agent_ids:
            logger.info(f"Updating agent {agent_name}")
            self.agents[agent_name].update()

    def save(self, path):
        for agent_name in self.agent_ids:
            self.agents[agent_name].save(path, filename=f"{agent_name}_ppo.pth")

    def load(self, path):
        for agent_name in self.agent_ids:
            self.agents[agent_name].load(path, filename=f"{agent_name}_ppo.pth")
