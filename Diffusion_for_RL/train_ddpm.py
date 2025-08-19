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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load("./data/stack_random_dataset_raw.npz")
    states = data["states"]
    actions = data["actions"]

    train_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)


    model = ConditionalDiffusionModel(state_dim=, action_dim=, time_emb_dim=128, hidden_dim=128)
    ddpm = ConditionalDDPM(model, device, T=1000)
    optimizer = torch.optim.AdamW(ddpm.model.parameters(), lr=2e-4, weight_decay=1e-4)

    ddpm.train(train_loader, optimizer, epochs=4000)



if __name__ == "__main__":
    main()