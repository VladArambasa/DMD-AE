import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # ENCODER MLP: (2d+2)-DIM STATE -> 512 -> 256 -> 128 -> LATENT
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)