"""Neg-Pearson correlation loss used by rPPG models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Neg_Pearson(nn.Module):
    """Neg-Pearson loss for rPPG signal reconstruction with NaN handling."""

    def __init__(self) -> None:
        """Initialize the base ``nn.Module``."""
        super().__init__()

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        """Compute `1 - Pearson(pred, label)` averaged over the batch."""
        loss = torch.zeros(1, device=preds.device)
        for i in range(preds.shape[0]):
            pred_i = preds[i]
            label_i = labels[i]
            if torch.isnan(pred_i).any() or torch.isnan(label_i).any():
                print("Warning: NaN values detected in inputs to Neg_Pearson loss")
                continue

            sum_x = torch.sum(pred_i)
            sum_y = torch.sum(label_i)
            sum_xy = torch.sum(pred_i * label_i)
            sum_x2 = torch.sum(torch.pow(pred_i, 2))
            sum_y2 = torch.sum(torch.pow(label_i, 2))
            N = pred_i.shape[0]

            # Add epsilon to avoid division by zero
            eps = 1e-8
            numerator = N * sum_xy - sum_x * sum_y
            denominator = torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2) + eps) *
                                     (N * sum_y2 - torch.pow(sum_y, 2) + eps))

            # Check for very small denominator
            if denominator < eps:
                print("Warning: Near-zero denominator in Pearson correlation")
                denominator = eps

            pearson = numerator / denominator

            # Clamp to valid range
            pearson = torch.clamp(pearson, min=-1.0, max=1.0)

            loss += 1 - pearson

        # If all batches had NaN, return a small positive value instead of NaN
        if preds.shape[0] == 0 or loss == 0:
            return torch.tensor(0.1, device=preds.device, requires_grad=True)

        loss = loss / preds.shape[0]

        # Final safety check
        if torch.isnan(loss):
            print("Warning: NaN loss produced in Neg_Pearson")
            return torch.tensor(0.1, device=preds.device, requires_grad=True)

        return loss.squeeze(0)
