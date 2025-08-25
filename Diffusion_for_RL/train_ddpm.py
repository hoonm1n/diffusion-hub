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
from torch.utils.data import TensorDataset, DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load("./data/stack_random_dataset_raw.npz")
    states = data["states"]
    actions = data["actions"]

    _, obs_dim = states.shape
    _, ac_dim = actions.shape

    states = torch.tensor(data["states"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)
    dataset = TensorDataset(states, actions)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)


    model = ConditionalDiffusionModel(state_dim=obs_dim, action_dim=ac_dim, time_emb_dim=128, hidden_dim=256)
    ddpm = ConditionalDDPM(model, device, T=1000)
    optimizer = torch.optim.AdamW(ddpm.model.parameters(), lr=2e-4, weight_decay=1e-4)

    ddpm.train(train_loader, optimizer, epochs=1000)



if __name__ == "__main__":
    main()