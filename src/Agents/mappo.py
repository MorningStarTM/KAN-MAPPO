from src.Agents.agent import PPO
from src.Utils.logger import logger
import torch

class MAPPO:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, env, config, scenario='default'):
        self.agents = {}
        self.agent_ids = env.possible_agents
        self.n_agents = n_agents

        for agent_idx, agent_name in enumerate(self.agent_ids):
            logger.info(f"Initializing PPO for {agent_name}")
            self.agents[agent_name] = PPO(
                state_dim=actor_dims[agent_name],
                action_dim=n_actions[agent_name],
                config=config
            )

        logger.info(f"MAPPO initialized with {self.n_agents} agents in scenario '{scenario}'")

    def select_action(self, observations):
        actions = {}
        logprobs = {}
        values = {}
        for agent_name in self.agent_ids:
            action, logprob, value = self.agents[agent_name].select_action(observations[agent_name])
            actions[agent_name] = action
            logprobs[agent_name] = logprob
            values[agent_name] = value
        return actions, logprobs, values

    def store_transition(self, agent_name, state, action, logprob, value, reward, done):
        agent = self.agents[agent_name]
        agent.buffer.states.append(state)
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
