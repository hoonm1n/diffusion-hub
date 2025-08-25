import torch
import matplotlib.pyplot as plt
import numpy as np
from ddpm import DDPM
from model import UNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(img_channels=3, base_channels=128, t_emb_dim=512).to(device)
    # model.load_state_dict(torch.load("./checkpoints/model_state_dict_5.pth", map_location=device))
    checkpoint = torch.load('./checkpoints/ddpm_checkpoint_1.pth', map_location=device)
    model.load_state_dict(checkpoint['ema_state_dict'])

    ddpm = DDPM(model, device, T=1000)

    x = ddpm.sample(image_size=32, batch_size=1)
    print(x)

    x_img = (x + 1) / 2 
    x_img = x_img.clamp(0, 1)

    img = x_img[0].permute(1, 2, 0).cpu().numpy()  # [C,H,W] -> [H,W,C]

    plt.imshow(img)
    plt.axis('off')
    plt.savefig("./sample/sample5.png")  
    plt.close()  






if __name__ == "__main__":
    main()