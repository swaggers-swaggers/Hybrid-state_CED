from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).square().mean()
    denominator = target.float().square().mean().clamp_min(1e-12)
    return numerator / denominator


def cosine_mean(prediction: torch.Tensor, target: torch.Tensor, head_dim: int) -> torch.Tensor:
    prediction = prediction.float().reshape(*prediction.shape[:-1], -1, head_dim)
    target = target.float().reshape(*target.shape[:-1], -1, head_dim)
    return F.cosine_similarity(prediction, target, dim=-1).mean()


class MetricAccumulator:
    def __init__(self) -> None:
        self.squared_error = defaultdict(float)
        self.target_energy = defaultdict(float)
        self.cosine_sum = defaultdict(float)
        self.count = defaultdict(int)
        self.batch_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def update_tensor(
        self,
        method: str,
        metric_prefix: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        head_dim: int,
    ) -> None:
        key = f"{method}:{metric_prefix}"
        pred = prediction.detach().float()
        tgt = target.detach().float()
        self.squared_error[key] += (pred - tgt).square().sum().item()
        self.target_energy[key] += tgt.square().sum().item()
        cosine = cosine_mean(pred, tgt, head_dim).item()
        batch_size = prediction.shape[0]
        self.cosine_sum[key] += cosine * batch_size
        self.count[key] += batch_size
        batch_nmse = ((pred - tgt).square().sum() / tgt.square().sum().clamp_min(1e-12)).item()
        self.batch_metrics[method][f"{metric_prefix}_nmse"].append(batch_nmse)

    def finalize(self, methods: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[float]]]]:
        output: dict[str, dict[str, float]] = {}
        for method in methods:
            metrics: dict[str, float] = {}
            for prefix in ("k", "v", "attention_output"):
                key = f"{method}:{prefix}"
                metrics[f"{prefix}_nmse"] = self.squared_error[key] / max(self.target_energy[key], 1e-12)
                metrics[f"{prefix}_cosine"] = self.cosine_sum[key] / max(self.count[key], 1)
            output[method] = metrics
        batches = {method: {name: values for name, values in metrics.items()} for method, metrics in self.batch_metrics.items()}
        return output, batches


def bootstrap_relative_improvement(
    baseline: list[float],
    candidate: list[float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if baseline_array.shape != candidate_array.shape or baseline_array.size == 0:
        raise ValueError("Paired bootstrap inputs must be non-empty and equally shaped")
    rng = np.random.default_rng(seed)
    improvements = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample_indices = rng.integers(0, baseline_array.size, baseline_array.size)
        base_mean = baseline_array[sample_indices].mean()
        candidate_mean = candidate_array[sample_indices].mean()
        improvements[index] = (base_mean - candidate_mean) / max(base_mean, 1e-12)
    return {
        "mean": float(improvements.mean()),
        "ci95_low": float(np.quantile(improvements, 0.025)),
        "ci95_high": float(np.quantile(improvements, 0.975)),
    }


def gate_decision(
    metrics: dict[str, dict[str, float]],
    batch_metrics: dict[str, dict[str, list[float]]],
    bootstrap_samples: int,
    minimum_point_improvement: float,
    minimum_ci_improvement: float,
    seed: int,
    selected_method: str | None = None,
) -> dict[str, Any]:
    baseline = metrics["original_projection"]
    trained_methods = [name for name in metrics if name not in {"original_projection", "random_linear", "zero"}]
    best_method = selected_method or min(trained_methods, key=lambda name: metrics[name]["attention_output_nmse"])
    if best_method not in trained_methods:
        raise ValueError(f"Selected method is not a trained probe: {best_method}")
    best = metrics[best_method]
    kv_baseline = (baseline["k_nmse"] + baseline["v_nmse"]) / 2
    kv_best = (best["k_nmse"] + best["v_nmse"]) / 2
    point_kv = (kv_baseline - kv_best) / max(kv_baseline, 1e-12)
    point_attention = (baseline["attention_output_nmse"] - best["attention_output_nmse"]) / max(
        baseline["attention_output_nmse"], 1e-12
    )
    attention_bootstrap = bootstrap_relative_improvement(
        batch_metrics["original_projection"]["attention_output_nmse"],
        batch_metrics[best_method]["attention_output_nmse"],
        bootstrap_samples,
        seed,
    )
    passed = (
        point_kv >= minimum_point_improvement
        and point_attention >= minimum_point_improvement
        and attention_bootstrap["ci95_low"] > minimum_ci_improvement
        and best["attention_output_nmse"] < metrics["random_linear"]["attention_output_nmse"]
    )
    return {
        "status": "GO" if passed else "NO-GO",
        "best_method": best_method,
        "kv_nmse_relative_improvement": point_kv,
        "attention_nmse_relative_improvement": point_attention,
        "attention_improvement_bootstrap": attention_bootstrap,
        "criteria": {
            "minimum_point_improvement": minimum_point_improvement,
            "minimum_ci_improvement": minimum_ci_improvement,
            "must_beat_random_attention": True,
        },
    }
