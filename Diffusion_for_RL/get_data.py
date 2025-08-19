import robosuite as suite
from robosuite.wrappers import GymWrapper
import gym
import gymnasium as gym
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from conditional_ddpm import ConditionalDDPM
from model import ConditionalDiffusionModel


def main():
    env = suite.make(
        env_name="Stack",
        robots="Panda",     
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        reward_shaping=True       
    )
    env = GymWrapper(env)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    states, actions = [], []

    num_samples = 400000
    max_steps = 250
    curr_step = 0

    while 1:
        state, _ = env.reset()
        done = False
        epi_step = 0
        while not done:
            action = env.action_space.sample()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # (s, a) 저장
            states.append(state)
            actions.append(action)

            state = next_state
            curr_step += 1
            epi_step += 1


            if epi_step >= max_steps:
                break
        
        print(curr_step)
        if curr_step >= num_samples:
            break

    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    np.savez("./data/stack_random_dataset_raw.npz",
         states=states,
         actions=actions)


if __name__ == "__main__":
    main()