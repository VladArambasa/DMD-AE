import torch
import numpy as np

class DMDDynamics:
    def __init__(self, device="cpu"):
        self.A = None
        self.device = device

    def fit(self, Z1, Z2, ridge: float = 1e-2):

        Z1_t = torch.tensor(Z1, dtype=torch.float32, device=self.device)
        Z2_t = torch.tensor(Z2, dtype=torch.float32, device=self.device)
        N = Z1_t.shape[0]
        G = (Z1_t.T @ Z1_t) / N
        H = (Z1_t.T @ Z2_t) / N
        eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        A_t = torch.linalg.solve(G + ridge * eye, H)
        self.A = A_t.T
        return self.A

    def predict(self, z0, steps=20):

        preds = []
        z = z0.clone().to(self.device)

        for _ in range(steps):
            if z.ndim == 1:
                z = self.A @ z
            else:
                z = z @ self.A.T
            preds.append(z.clone())

        return torch.stack(preds)