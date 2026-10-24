"""Temperature Scaling Calibration and Expected Calibration Error (ECE) Utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_multiclass_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) for multiclass probability predictions.

    Args:
        probs: (N, C) normalized probability distribution.
        labels: (N,) ground-truth class indices.
        num_bins: Number of confidence intervals in [0, 1].
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece.item())


def compute_binary_ece(
    probs_pos: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) for binary `noul` probabilities."""
    probs_pos = probs_pos.view(-1)
    labels = labels.view(-1).long()
    probs_2c = torch.stack([1.0 - probs_pos, probs_pos], dim=-1)
    return compute_multiclass_ece(probs_2c, labels, num_bins=num_bins)


class TemperatureScaler(nn.Module):
    """Post-hoc Temperature Scaling module for calibrating Laya's 3 decision heads."""

    def __init__(self) -> None:
        super().__init__()
        self.log_t_maneuver = nn.Parameter(torch.zeros(1))
        self.log_t_speed = nn.Parameter(torch.zeros(1))
        self.log_t_aeb = nn.Parameter(torch.zeros(1))
        self.b_aeb = nn.Parameter(torch.zeros(1))

    @property
    def t_maneuver(self) -> torch.Tensor:
        return torch.exp(self.log_t_maneuver).clamp(min=0.05, max=20.0)

    @property
    def t_speed(self) -> torch.Tensor:
        return torch.exp(self.log_t_speed).clamp(min=0.05, max=20.0)

    @property
    def t_aeb(self) -> torch.Tensor:
        return torch.exp(self.log_t_aeb).clamp(min=0.05, max=20.0)

    def get_temperatures(self) -> Dict[str, float]:
        return {
            "t_maneuver": float(self.t_maneuver.item()),
            "t_speed": float(self.t_speed.item()),
            "t_aeb": float(self.t_aeb.item()),
            "b_aeb": float(self.b_aeb.item()),
        }

    @staticmethod
    def _optimize_single_multiclass_temp(
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 15,
    ) -> float:
        """Find optimal scalar temperature T minimizing ECE + 0.02 * NLL on validation logits."""
        best_t = 1.0
        best_obj = float("inf")
        grid = torch.linspace(0.05, 2.50, 400, device=logits.device)
        for t_cand in grid:
            t_val = float(t_cand.item())
            scaled = logits / t_val
            probs = F.softmax(scaled, dim=-1)
            ece = compute_multiclass_ece(probs, targets, num_bins=num_bins)
            obj = ece
            if obj < best_obj:
                best_obj = obj
                best_t = t_val
        return best_t

    @staticmethod
    def _optimize_single_binary_temp(
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 15,
    ) -> Tuple[float, float]:
        best_t = 1.0
        best_b = 0.0
        best_obj = float("inf")
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).float()
        t_grid = torch.logspace(-1.0, 1.2, 80, device=logits.device)
        if float(targets_flat.sum().item()) == 0.0:
            b_grid = torch.zeros(1, device=logits.device)
        else:
            b_grid = torch.linspace(-4.0, 2.0, 45, device=logits.device)
        for t_cand in t_grid:
            t_val = float(t_cand.item())
            for b_cand in b_grid:
                b_val = float(b_cand.item())
                scaled = (logits_flat + b_val) / t_val
                nll = float(
                    F.binary_cross_entropy_with_logits(scaled, targets_flat).item()
                )
                probs = torch.sigmoid(scaled)
                ece = compute_binary_ece(probs, targets_flat, num_bins=num_bins)
                obj = ece + 0.02 * nll
                if obj < best_obj:
                    best_obj = obj
                    best_t = t_val
                    best_b = b_val
        return best_t, best_b

    def fit(
        self,
        maneuver_logits: torch.Tensor,
        maneuver_targets: torch.Tensor,
        speed_logits: torch.Tensor,
        speed_targets: torch.Tensor,
        aeb_logits: torch.Tensor,
        aeb_targets: torch.Tensor,
        num_bins: int = 15,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Optimize temperatures on validation logits and return (before_metrics, after_metrics)."""
        with torch.no_grad():
            before_m_ece = compute_multiclass_ece(
                F.softmax(maneuver_logits, dim=-1), maneuver_targets, num_bins=num_bins
            )
            before_s_ece = compute_multiclass_ece(
                F.softmax(speed_logits, dim=-1), speed_targets, num_bins=num_bins
            )
            before_a_ece = compute_binary_ece(
                torch.sigmoid(aeb_logits), aeb_targets, num_bins=num_bins
            )
            before_mean_ece = (before_m_ece + before_s_ece + before_a_ece) / 3.0

            t_m = self._optimize_single_multiclass_temp(
                maneuver_logits, maneuver_targets, num_bins=num_bins
            )
            t_s = self._optimize_single_multiclass_temp(
                speed_logits, speed_targets, num_bins=num_bins
            )
            t_a, b_a = self._optimize_single_binary_temp(
                aeb_logits, aeb_targets, num_bins=num_bins
            )

            self.log_t_maneuver.data.fill_(float(torch.log(torch.tensor(t_m)).item()))
            self.log_t_speed.data.fill_(float(torch.log(torch.tensor(t_s)).item()))
            self.log_t_aeb.data.fill_(float(torch.log(torch.tensor(t_a)).item()))
            self.b_aeb.data.fill_(float(b_a))

            after_m_ece = compute_multiclass_ece(
                F.softmax(maneuver_logits / self.t_maneuver, dim=-1),
                maneuver_targets,
                num_bins=num_bins,
            )
            after_s_ece = compute_multiclass_ece(
                F.softmax(speed_logits / self.t_speed, dim=-1),
                speed_targets,
                num_bins=num_bins,
            )
            after_a_ece = compute_binary_ece(
                torch.sigmoid((aeb_logits + self.b_aeb) / self.t_aeb),
                aeb_targets,
                num_bins=num_bins,
            )
            after_mean_ece = (after_m_ece + after_s_ece + after_a_ece) / 3.0

        before_metrics = {
            "maneuver_ece": before_m_ece,
            "speed_ece": before_s_ece,
            "aeb_ece": before_a_ece,
            "mean_ece": before_mean_ece,
        }
        after_metrics = {
            "maneuver_ece": after_m_ece,
            "speed_ece": after_s_ece,
            "aeb_ece": after_a_ece,
            "mean_ece": after_mean_ece,
            **self.get_temperatures(),
        }
        return before_metrics, after_metrics
