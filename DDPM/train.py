import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from ddpm import DDPM
from model import UNet


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(img_channels=3, base_channels=128, t_emb_dim=128)
    ddpm = DDPM(model, device, T=1000)
    optimizer = torch.optim.AdamW(ddpm.model.parameters(), lr=2e-4, weight_decay=1e-4)

    ddpm.train(train_loader, optimizer, epochs=4000)



if __name__ == "__main__":
    main()