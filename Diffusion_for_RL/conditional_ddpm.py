import numpy as np
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils as nn_utils

from model import ConditionalDiffusionModel

from torch.distributions import Normal



writer = SummaryWriter(log_dir=f"runs/ConditionalDDPM_{int(time.time())}")

class ConditionalDDPM:
    def __init__(self, model, device, T=1000):
        self.device = device        
        self.model = model.to(device)
        self.T = T

        self.betas = torch.linspace(1e-4, 0.02, self.T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)




    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)



    def compute_x_noisy(self, x, t, noise):
        t = t.long()
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1)
        x_t = sqrt_alpha_cumprod_t * x + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t




    def train(self, dataloader, optimizer, epochs):
        self.model.train()
        total_step = 0
        for epoch in range(epochs):
            for step, (state, action) in enumerate(dataloader):
                state = state.to(self.device)
                action = action.to(self.device)
                batch_size = action.size(0)
                t = self.sample_timesteps(batch_size)

                noise = torch.randn_like(action)

                action_noisy = self.compute_x_noisy(action, t, noise)

                noise_pred = self.model(action_noisy, state, t)

                loss = nn.MSELoss()(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_step += 1

                if step % 100 == 0:
                    print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
                    writer.add_scalar("Loss", loss.item(), total_step)


            torch.save(self.model.state_dict(), './checkpoints/ddpm_pretrain_1.pth')
        torch.save(self.model.state_dict(), './checkpoints/ddpm_pretrain_1.pth')           
        


    @torch.no_grad()
    def sample(self, states):
        self.model.eval()
        batch_size, state_dim = states.shape
        x = torch.randn(batch_size, self.model.net[-1].out_features).to(self.device)   # action_dim

        for t in reversed(range(self.T)):
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            beta_t = self.betas[t].view(-1,1)
            alpha_t = self.alphas[t].view(-1,1)
            alpha_cumprod_t = self.alpha_cumprod[t].view(-1,1)

            pred_noise = self.model(x, states, t_tensor)
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t)/torch.sqrt(1 - alpha_cumprod_t)) * pred_noise) + torch.sqrt(beta_t) * z
        return x






