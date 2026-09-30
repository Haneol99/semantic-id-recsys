"""RQ-VAE for Semantic IDs (TIGER, Rajput et al. 2023, Sec. 3.1).

Encoder MLP 768 -> 512 -> 256 -> 128 -> 32 (ReLU between layers), mirrored decoder, and a residual quantizer
with `levels` codebooks of `codebook_size` x `dim`. At level d the residual r_d is replaced by its nearest code
e_{c_d}, and r_{d+1} = r_d - e_{c_d} (r_0 = encoder output). The decoder sees z + sg(sum_d e_{c_d} - z)
(straight-through estimator).

Loss = MSE(x_hat, x) + sum_d [ MSE(sg(r_d), e_{c_d}) + beta * MSE(r_d, sg(e_{c_d})) ].

Codebooks are initialized with k-means on the first training batch, level by level (level d on that batch's
residuals r_d after levels < d are initialized).

Deviations from the paper, both optional (see configs/rqvae.yaml vs configs/rqvae_paper.yaml):
- input standardization: x is standardized per dimension with stored mean/std before the encoder, and the
  reconstruction target is the standardized x (unit-norm Sentence-T5 vectors otherwise collapse to one code);
- dead-code reset: `reset_dead_codes` moves every unused code at each level onto a random current residual.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn


def mlp(dims: Sequence[int]) -> nn.Sequential:
    """Linear layers with ReLU between them (none after the last)."""
    layers = []
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ResidualQuantizer(nn.Module):
    def __init__(self, levels: int, codebook_size: int, dim: int):
        super().__init__()
        self.codebooks = nn.Parameter(torch.randn(levels, codebook_size, dim) * 0.1)

    @property
    def levels(self) -> int:
        return self.codebooks.shape[0]

    @staticmethod
    def nearest(r: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Index of the nearest code (squared Euclidean) for each row of r."""
        d = r.pow(2).sum(1, keepdim=True) - 2 * r @ codebook.T + codebook.pow(2).sum(1)[None]
        return d.argmin(1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (codes (B, levels), quantized sum (B, dim), (codebook_loss, commit_loss))."""
        residual, quantized = z, torch.zeros_like(z)
        codes, codebook_loss, commit_loss = [], 0.0, 0.0
        for d in range(self.levels):
            idx = self.nearest(residual.detach(), self.codebooks[d])
            e = self.codebooks[d][idx]
            codebook_loss = codebook_loss + F.mse_loss(e, residual.detach())
            commit_loss = commit_loss + F.mse_loss(residual, e.detach())
            codes.append(idx)
            quantized = quantized + e
            residual = residual - e.detach()
        return torch.stack(codes, 1), quantized, (codebook_loss, commit_loss)

    @torch.no_grad()
    def reset_dead_codes(self, z: torch.Tensor, generator: torch.Generator) -> list[int]:
        """Per level, move codes no row of z maps to onto randomly chosen residuals; returns #reset per level."""
        residual, n_reset = z.detach(), []
        k = self.codebooks.shape[1]
        for d in range(self.levels):
            idx = self.nearest(residual, self.codebooks[d])
            dead = (torch.bincount(idx, minlength=k) == 0).nonzero().squeeze(1)
            if len(dead):
                pick = torch.randperm(len(residual), generator=generator)[: len(dead)].to(residual.device)
                self.codebooks.data[d, dead] = residual[pick]
                idx = self.nearest(residual, self.codebooks[d])
            residual = residual - self.codebooks[d][idx]
            n_reset.append(int(len(dead)))
        return n_reset

    @torch.no_grad()
    def kmeans_init(self, z: torch.Tensor, seed: int) -> None:
        residual = z.detach().cpu().numpy().astype(np.float64)
        k = self.codebooks.shape[1]
        for d in range(self.levels):
            km = KMeans(n_clusters=k, n_init=1, random_state=seed + d).fit(residual)
            centers = km.cluster_centers_
            self.codebooks[d] = torch.as_tensor(centers, dtype=self.codebooks.dtype, device=self.codebooks.device)
            residual = residual - centers[km.labels_]


class RQVAE(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dims: Sequence[int] = (512, 256, 128), latent_dim: int = 32,
                 levels: int = 3, codebook_size: int = 256, beta: float = 0.25):
        super().__init__()
        self.beta = beta
        self.encoder = mlp([input_dim, *hidden_dims, latent_dim])
        self.decoder = mlp([latent_dim, *reversed(hidden_dims), input_dim])
        self.quantizer = ResidualQuantizer(levels, codebook_size, latent_dim)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))

    @torch.no_grad()
    def set_input_standardization(self, x: torch.Tensor) -> None:
        """Standardize inputs per dimension with these statistics (identity by default)."""
        self.input_mean.copy_(x.mean(0))
        self.input_std.copy_(x.std(0).clamp_min(1e-8))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - self.input_mean) / self.input_std
        z = self.encoder(x)
        codes, quantized, (codebook_loss, commit_loss) = self.quantizer(z)
        x_hat = self.decoder(z + (quantized - z).detach())
        recon_loss = F.mse_loss(x_hat, x)
        rq_loss = codebook_loss + self.beta * commit_loss
        return {"codes": codes, "loss": recon_loss + rq_loss, "recon_loss": recon_loss, "rq_loss": rq_loss}

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder((x - self.input_mean) / self.input_std)

    @torch.no_grad()
    def kmeans_init(self, x: torch.Tensor, seed: int) -> None:
        self.quantizer.kmeans_init(self.latent(x), seed)

    @torch.no_grad()
    def reset_dead_codes(self, x: torch.Tensor, generator: torch.Generator) -> list[int]:
        return self.quantizer.reset_dead_codes(self.latent(x), generator)

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        """Codes (N, levels) for every row of x."""
        return torch.cat([self.quantizer(self.latent(x[i:i + batch_size]))[0] for i in range(0, len(x), batch_size)])
