from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn


class FullRankKVProbe(nn.Module):
    def __init__(self, hidden_size: int, kv_size: int) -> None:
        super().__init__()
        self.k_proj = nn.Linear(hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_size, bias=False)

    def forward(self, sources: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = sources[max(sources)]
        return self.k_proj(hidden), self.v_proj(hidden)


class LowRankLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(input_size, rank, bias=False)
        self.up = nn.Linear(rank, output_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden))


class LowRankKVProbe(nn.Module):
    def __init__(self, hidden_size: int, kv_size: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.k_proj = LowRankLinear(hidden_size, kv_size, rank)
        self.v_proj = LowRankLinear(hidden_size, kv_size, rank)

    def forward(self, sources: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = sources[max(sources)]
        return self.k_proj(hidden), self.v_proj(hidden)


class AsymmetricFusionKVProbe(nn.Module):
    def __init__(self, hidden_size: int, kv_size: int, source_layers: Iterable[int]) -> None:
        super().__init__()
        self.source_layers = tuple(source_layers)
        if len(self.source_layers) < 2:
            raise ValueError("Asymmetric fusion needs at least two source layers")
        self.k_logits = nn.Parameter(torch.zeros(len(self.source_layers)))
        self.v_logits = nn.Parameter(torch.zeros(len(self.source_layers)))
        self.k_proj = nn.Linear(hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_size, bias=False)

    def fusion_weights(self) -> dict[str, list[float]]:
        return {
            "k": torch.softmax(self.k_logits.detach().float(), dim=0).cpu().tolist(),
            "v": torch.softmax(self.v_logits.detach().float(), dim=0).cpu().tolist(),
        }

    def forward(self, sources: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.stack([sources[index] for index in self.source_layers], dim=0)
        k_weights = torch.softmax(self.k_logits, dim=0).to(hidden.dtype).view(-1, 1, 1, 1)
        v_weights = torch.softmax(self.v_logits, dim=0).to(hidden.dtype).view(-1, 1, 1, 1)
        k_source = (hidden * k_weights).sum(dim=0)
        v_source = (hidden * v_weights).sum(dim=0)
        return self.k_proj(k_source), self.v_proj(v_source)


def build_trainable_probes(
    hidden_size: int,
    kv_size: int,
    source_layers: Iterable[int],
    low_ranks: Iterable[int],
) -> nn.ModuleDict:
    probes: OrderedDict[str, nn.Module] = OrderedDict()
    probes["trained_linear"] = FullRankKVProbe(hidden_size, kv_size)
    for rank in low_ranks:
        probes[f"low_rank_{rank}"] = LowRankKVProbe(hidden_size, kv_size, rank)
    probes["multi_layer_fusion"] = AsymmetricFusionKVProbe(hidden_size, kv_size, source_layers)
    return nn.ModuleDict(probes)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
