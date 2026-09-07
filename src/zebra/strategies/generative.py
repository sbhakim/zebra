"""Temporal VAE generative replay with previous-solver distillation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from zebra.model import N_FEATURES
from zebra.strategies.base import Strategy


class TemporalVAE(nn.Module):
    """Small sequence VAE operating on genuine ``(time, feature)`` windows."""

    def __init__(self, hidden: int = 32, latent: int = 16, length: int = 10) -> None:
        super().__init__()
        self.hidden = hidden
        self.latent = latent
        self.length = length
        self.encoder = nn.LSTM(N_FEATURES, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.initial = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(N_FEATURES, hidden, batch_first=True)
        self.output = nn.Linear(hidden, N_FEATURES)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(x)
        final = encoded[:, -1]
        return self.mu(final), self.logvar(final)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.initial(z)).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        inputs = torch.zeros(
            (len(z), self.length, N_FEATURES), device=z.device, dtype=z.dtype
        )
        decoded, _ = self.decoder(inputs, (hidden, cell))
        output: torch.Tensor = self.output(decoded)
        return output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


class GenerativeReplay(Strategy):
    name = "gen-replay"

    def __init__(
        self,
        model: nn.Module,
        replay_ratio: float = 0.5,
        temperature: float = 2.0,
        vae_epochs: int = 5,
        vae_lr: float = 1e-3,
    ) -> None:
        super().__init__(model)
        self.replay_ratio = replay_ratio
        self.temperature = temperature
        self.vae_epochs = vae_epochs
        self.vae_lr = vae_lr
        self.generator_model: TemporalVAE | None = None
        self.teacher: nn.Module | None = None

    @property
    def retained(self) -> Any:
        return {"generator": self.generator_model, "teacher": self.teacher}

    def penalty(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        if self.generator_model is None or self.teacher is None:
            return torch.zeros((), device=logits.device)
        n = max(1, round(self.replay_ratio * len(x)))
        with torch.no_grad():
            z = torch.randn((n, self.generator_model.latent), device=x.device)
            synthetic = self.generator_model.decode(z)
            teacher_logits = self.teacher(synthetic)
            teacher_prob = torch.nn.functional.softmax(
                teacher_logits / self.temperature, dim=1
            )
        student_logits = self.model(synthetic)
        student_log = torch.nn.functional.log_softmax(
            student_logits / self.temperature, dim=1
        )
        return (1.0 - self.replay_ratio) * self.temperature**2 * torch.nn.functional.kl_div(
            student_log, teacher_prob, reduction="batchmean"
        )

    def after_domain(self, loader: Any, device: torch.device) -> None:
        current = torch.cat([batch[0] for batch in loader], dim=0)
        if self.generator_model is not None:
            with torch.no_grad():
                z = torch.randn((len(current), self.generator_model.latent), device=device)
                previous = self.generator_model.to(device).decode(z).cpu()
            current = torch.cat((current, previous), dim=0)
        vae = TemporalVAE(length=current.shape[1]).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=self.vae_lr)
        data = DataLoader(TensorDataset(current), batch_size=128, shuffle=True)
        vae.train()
        for _ in range(self.vae_epochs):
            for (batch,) in data:
                batch = batch.to(device)
                reconstructed, mu, logvar = vae(batch)
                reconstruction = torch.nn.functional.mse_loss(reconstructed, batch)
                kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
                loss = reconstruction + 1e-3 * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
        self.generator_model = vae.eval()
        for parameter in self.generator_model.parameters():
            parameter.requires_grad_(False)
        self.teacher = deepcopy(self.model).to(device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
