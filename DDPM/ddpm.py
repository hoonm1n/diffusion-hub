import numpy as np
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils as nn_utils

from model import UNet

from torch.distributions import Normal
import copy



writer = SummaryWriter(log_dir=f"runs/ddpm_{int(time.time())}")

class DDPM:
    def __init__(self, model, device, T=1000, ema_decay=0.9999):
        self.device = device        
        self.model = model.to(device)
        self.T = T

        self.betas = torch.linspace(1e-4, 0.02, self.T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)



        self.ema_decay = ema_decay
        self.ema_model = copy.deepcopy(self.model).eval()  
        for param in self.ema_model.parameters():
            param.requires_grad = False




    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)



    def compute_x_noisy(self, x, t, noise):
        t = t.long() 
        sqrt_alphas_cumprod_t = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1, 1)

        x_t = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t




    def train(self, dataloader, optimizer, epochs):
        self.model.train()
        total_step = 0
        for epoch in range(epochs):
            for step, (x, _) in enumerate(dataloader):
                x = x.to(self.device)
                batch_size = x.size(0)
                t = self.sample_timesteps(batch_size)

                noise = torch.randn_like(x)

                x_noisy = self.compute_x_noisy(x, t, noise)

                noise_pred = self.model(x_noisy, t)

                loss = nn.MSELoss()(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                

                self.update_ema()



                total_step += 1

                if step % 100 == 0:
                    print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
                    writer.add_scalar("Loss", loss.item(), total_step)


            torch.save({'model_state_dict': self.model.state_dict(),'ema_state_dict': self.ema_model.state_dict(),'step': total_step}, './checkpoints/ddpm_checkpoint_1.pth')
        torch.save({'model_state_dict': self.model.state_dict(),'ema_state_dict': self.ema_model.state_dict(),'step': total_step}, './checkpoints/ddpm_checkpoint_1.pth')          
        





    @torch.no_grad()
    def sample(self, image_size, batch_size=1, use_ema=True):
        model = self.ema_model if use_ema else self.model
        model.eval()

        x = torch.randn((batch_size, 3, image_size, image_size), device=self.device)
        for t in reversed(range(self.T)):
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            time_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            beta_t = self.betas[t].view(-1,1,1,1)
            alpha_t = self.alphas[t].view(-1,1,1,1)
            alpha_cumprod_t = self.alpha_cumprod[t].view(-1,1,1,1)

            pred_noise = model(x, time_tensor)
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1-alpha_t)/torch.sqrt(1-alpha_cumprod_t))*pred_noise) + torch.sqrt(beta_t)*z

        x = torch.clamp(x, -1, 1)
        return x







    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data * (1 - self.ema_decay))





