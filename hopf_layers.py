"""Hopf Latent Space projectors for structured-task evaluation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def hopf_fibration(q):
    """Map unit quaternions [..., 4] to S^2 base and S^1 fiber (cos, sin)."""
    w, x, y, z = q.unbind(-1)
    base = torch.stack([
        2.0 * (w * x + y * z),
        2.0 * (w * y - z * x),
        w * w - x * x - y * y + z * z,
    ], dim=-1)
    base = F.normalize(base, dim=-1, eps=1e-6)
    phase = torch.atan2(y, w)
    fiber = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
    return base, fiber


class HopfLatentProjector(nn.Module):
    """Nested Hopf fibration projector: S^1 fibers over CP^n bases."""

    def __init__(self, dim, num_levels=2):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.quat_dim = num_levels * 4
        self.in_proj = nn.Linear(dim, self.quat_dim, bias=False)
        self.out_proj = nn.Linear(num_levels * 5, dim, bias=False)
        self.level_gates = nn.Parameter(torch.zeros(num_levels))

    def forward(self, x):
        quats = self.in_proj(x).view(*x.shape[:-1], self.num_levels, 4)
        quats = F.normalize(quats, dim=-1, eps=1e-6)
        bases, fibers = [], []
        for level in range(self.num_levels):
            q = quats[..., level, :]
            gate = torch.sigmoid(self.level_gates[level])
            q = F.normalize(gate * q + (1.0 - gate) * quats[..., level, :], dim=-1, eps=1e-6)
            base, fiber = hopf_fibration(q)
            bases.append(base)
            fibers.append(fiber)
        structured = torch.cat(bases + fibers, dim=-1)
        return self.out_proj(structured)


class HypersphereProjector(nn.Module):
    """Ablation baseline: L2-normalize to unit sphere without Hopf structure."""

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.scale * F.normalize(x, dim=-1, eps=1e-6)


def latent_fiber_phase_variance(z, num_levels=2):
    """Periodicity proxy: mean variance of Hopf fiber phases derived from latent."""
    quat_dim = num_levels * 4
    if z.shape[-1] < quat_dim:
        z = F.pad(z, (0, quat_dim - z.shape[-1]))
    quats = z[..., :quat_dim].reshape(*z.shape[:-1], num_levels, 4)
    quats = F.normalize(quats, dim=-1, eps=1e-6)
    variances = []
    for level in range(num_levels):
        _, fiber = hopf_fibration(quats[..., level, :])
        phase = torch.atan2(fiber[..., 1], fiber[..., 0])
        variances.append(phase.var().item())
    return float(sum(variances) / len(variances))


def hopf_fiber_phase_variance(z, projector: HopfLatentProjector):
    """Periodicity proxy using a trained HopfLatentProjector's quaternion map."""
    with torch.no_grad():
        quats = projector.in_proj(z).view(*z.shape[:-1], projector.num_levels, 4)
        quats = F.normalize(quats, dim=-1, eps=1e-6)
        variances = []
        for level in range(projector.num_levels):
            q = quats[..., level, :]
            gate = torch.sigmoid(projector.level_gates[level])
            q = F.normalize(gate * q + (1.0 - gate) * quats[..., level, :], dim=-1, eps=1e-6)
            _, fiber = hopf_fibration(q)
            phase = torch.atan2(fiber[..., 1], fiber[..., 0])
            variances.append(phase.var().item())
    return float(sum(variances) / len(variances))