import torch
import matplotlib.pyplot as plt
import numpy as np
from conditional_ddpm import ConditionalDDPM
from model import ConditionalDiffusionModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load("./data/stack_random_dataset_raw.npz")
    states = data["states"]
    actions = data["actions"]

    _, obs_dim = states.shape
    _, ac_dim = actions.shape


    model = ConditionalDiffusionModel(state_dim=obs_dim, action_dim=ac_dim, time_emb_dim=128, hidden_dim=256).to(device)
    model.load_state_dict(torch.load("./checkpoints/ddpm_pretrain_1.pth", map_location=device))

    ddpm = ConditionalDDPM(model, device, T=1000)

    print(states.shape)
    obs = states[0].reshape(1,-1)
    print(obs.shape)
    x = ddpm.sample(obs)
    print(x)
    print(x.shape)








if __name__ == "__main__":
    main()