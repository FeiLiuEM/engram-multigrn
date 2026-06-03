"""
Context-aware gating module and ShortConv.
Adapted from DeepSeek Engram for gene regulatory context gating.
"""
import math
import torch
import torch.nn as nn


class ContextGate(nn.Module):
    """
    Context-aware gating mechanism.
    Query: cell condition embedding (e.g., lactate level, cell type)
    Key: retrieved regulatory memory from hash lookup
    Gate: sigmoid(normalized_query · normalized_key / sqrt(d))
    """
    def __init__(self, condition_dim: int, memory_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.condition_norm = nn.RMSNorm(hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.memory_norm = nn.RMSNorm(hidden_dim)
        self.sqrt_d = math.sqrt(hidden_dim)

    def forward(
        self,
        memory: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        memory: [B, D_mem] or [B, N, D_mem] - retrieved regulatory memory
        condition: [B, D_cond] - cell condition vector
        Returns: [B, D_mem] or [B, N, D_mem] - gated memory
        """
        q = self.condition_norm(self.condition_proj(condition))
        k = self.memory_norm(self.memory_proj(memory))
        ndim = memory.dim()
        if ndim == 3:
            q = q.unsqueeze(1)
        scores = (q * k).sum(dim=-1, keepdim=True) / self.sqrt_d
        gate = scores.abs().clamp_min(1e-6).sqrt() * scores.sign()
        gate = gate.sigmoid()
        return memory * gate


class ShortConv(nn.Module):
    """
    Depthwise 1D convolution for local regulatory refinement.
    Adapted from Engram's ShortConv.
    """
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        dilation: int = 2,
        hc_mult: int = 2,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)
        ])
        self.act_fn = nn.SiLU()
        self.final_norm = nn.RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, hc_mult, D] or [B, L, D]
        If no hc_mult dim, we add it.
        """
        if x.dim() == 3:
            x = x.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        B, T, G, C = x.shape
        normed = torch.cat(
            [self.norms[i](x[:, :, i, :]) for i in range(G)], dim=-1
        )
        x_bct = normed.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]
        y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        y = y.mean(dim=2)
        y = self.final_norm(y)
        return y
