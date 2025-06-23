from pettingzoo.mpe import simple_adversary_v3
from src.Agents.mappo import MAPPO
import numpy as np
import time


def test_mappo_interaction_visual():
    # Setup parallel env with rendering
    env = simple_adversary_v3.parallel_env(render_mode="human")
    env.reset()

    agents = env.possible_agents
    actor_dims = {agent: env.observation_space(agent).shape[0] for agent in agents}
    n_actions = {agent: env.action_space(agent).n for agent in agents}
    critic_dims = sum(actor_dims.values())

    # PPO config
    config = {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'eps_clip': 0.2,
        'K_epochs': 4,
    }

    mappo = MAPPO(actor_dims, critic_dims, len(agents), n_actions, env, config)

    # Reset the environment
    obs, _ = env.reset()
    step = 0

    while step < 20 and env.agents:
        actions, logprobs, values = mappo.select_action(obs)

        print(f"\nStep {step + 1}")
        for agent in actions:
            print(f"{agent}: action={actions[agent]}, logprob={logprobs[agent]:.4f}, value={values[agent].item():.4f}")

        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Exit loop if all agents are done
        if all(terminations[a] or truncations[a] for a in env.agents):
            print("Episode ended.")
            break

        step += 1
        time.sleep(0.2)  # Optional: slow down rendering a bit

    env.close()


if __name__ == "__main__":
    test_mappo_interaction_visual()
