import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb



class ConditionalDiffusionModel(nn.Module):
    def __init__(self, state_dim, action_dim, time_emb_dim=128, hidden_dim=128):
        super().__init__()
        self.timestep_embedding = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim+time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.noise_a_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, action_dim),  
        )
    
    def forward(self, noisy_a, s_t, t):
        t_emb = self.timestep_embedding(t)
        t_emb = self.time_mlp(t_emb)

        state_emb = self.state_mlp(s_t)                   
        cond = torch.cat([state_emb, t_emb], dim=-1)          
        cond_emb = self.cond_mlp(cond) 

        noisy_a_emb = self.noise_a_mlp(noisy_a)

        x = torch.cat([noisy_a_emb, cond_emb], dim=-1)
        noise_pred = self.net(x)                       
        return noise_pred