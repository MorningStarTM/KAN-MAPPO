import os
import numpy as np
from PIL import Image
import imageio
import torch
from src.Agents.mappo import MAPPO
from pettingzoo.mpe import simple_spread_v3

def evaluate_mappo_and_save_gif(
    model_dir, 
    scenario='simple_spread_v3', 
    n_episodes=5, 
    max_ep_len=25, 
    gif_save_path='eval_results.gif',
    config=None
):
    # Set up env
    env = simple_spread_v3.parallel_env(max_cycles=max_ep_len, continuous_actions=False, render_mode="rgb_array")
    env.reset()
    agents = env.possible_agents
    n_agents = len(agents)

    # Dimensions (should match your training setup)
    state_dims = {agent: env.observation_space(agent).shape[0] for agent in agents}
    action_dims = {agent: env.action_space(agent).n for agent in agents}

    # Load MAPPO with trained weights
    mappo = MAPPO(state_dims, n_agents, action_dims, env, config, scenario)
    mappo.load(model_dir)

    all_frames = []
    all_episode_rewards = {agent: [] for agent in agents}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_rewards = {agent: 0.0 for agent in agents}
        frames = []

        for t in range(max_ep_len):
            # Deterministic/eval action
            actions, _, _ = mappo.select_action(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            # Collect frame
            frame = env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))
            else:
                # Sometimes env.render() returns a PIL image already
                frames.append(frame)
            # Update rewards
            for agent in agents:
                episode_rewards[agent] += rewards[agent]
            # Check if all agents done
            obs = next_obs
            if all(terminations[a] or truncations[a] for a in agents):
                break

        # Save episode frames
        if frames:
            all_frames.extend(frames)
            all_frames.append(Image.new("RGB", frames[0].size, (255, 255, 255)))
        # Store rewards
        for agent in agents:
            all_episode_rewards[agent].append(episode_rewards[agent])

        print(f"Episode {ep+1}: " + ", ".join([f"{a}: {episode_rewards[a]:.2f}" for a in agents]))

    # Save GIF
    all_frames[0].save(
        gif_save_path,
        save_all=True,
        append_images=all_frames[1:],
        duration=80,  # ms per frame
        loop=0
    )
    print(f"Saved GIF to {gif_save_path}")
    return all_episode_rewards

# Usage:
if __name__ == '__main__':
    # Adjust config if needed (must match your training config for network architecture)
    config = {
            'lr_actor': 1e-4,
            'lr_critic': 1e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'K_epochs': 80,
            'update_timestep': 512,
            'n_episodes': 1000,
            'max_training_timesteps': int(3e6),
            'max_ep_len': 1000,
            'print_freq': 1000 * 10,
            'save_model_freq': int(1e5),
            'random_seed': 42,
        }
    evaluate_mappo_and_save_gif(
        model_dir='models/simple_spread_v3',  # <- path to your saved weights
        scenario='simple_spread_v3',
        n_episodes=5,
        max_ep_len=25,
        gif_save_path='mappo_eval.gif',
        config=config
    )
