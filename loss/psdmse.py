"""Power Spectral Density MSE loss utilities."""

from __future__ import annotations

import torch
import torch.fft
import torch.nn.functional as F
from torch import Tensor, nn


class NormPSD(nn.Module):
    """
    Power Spectral Density MSE loss.
    Calculates normalized PSD for output and target. MSE on normalized PSDs.
    NormPSD code gently borrowed from https://github.com/ToyotaResearchInstitute/RemotePPG
    """

    def __init__(self, high_pass: float, low_pass: float) -> None:
        """Store passband boundaries."""
        super().__init__()
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x: Tensor, fs: int, zero_pad: float = 0) -> Tensor:
        """Return normalized PSD of `x` within the configured band."""
        # Handle NaN input
        if torch.isnan(x).any():
            print("Warning: NaN input to NormPSD")
            # Replace NaNs with zeros
            x = torch.nan_to_num(x, nan=0.0)
        
        # Mean-center the signal
        x = x - torch.mean(x, dim=-1, keepdim=True)
        
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)
        
        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, :, 0] **2, x[:, :, 1] **2)
        
        # Filter PSD for relevant parts
        Fn = fs / 2
        freqs = torch.linspace(0, Fn, x.shape[1], device=x.device)
        use_freqs = torch.logical_and(freqs <= self.high_pass, freqs >= self.low_pass)
        
        # Handle case where no frequencies are selected
        if not use_freqs.any():
            print("Warning: No frequencies selected in PSD calculation")
            # Select at least one frequency
            use_freqs[0] = True
        
        x = x[:, use_freqs]
        
        # Safe normalization with epsilon
        eps = 1e-8
        sum_x = torch.sum(x, dim=-1, keepdim=True)
        # If sum is 0 or very small, add epsilon to avoid division by zero
        sum_x = torch.where(sum_x > eps, sum_x, torch.ones_like(sum_x) * eps)
        x = x / sum_x
        
        # Final safety check
        if torch.isnan(x).any():
            print("Warning: NaN output from NormPSD")
            x = torch.nan_to_num(x, nan=0.0)
        
        return x


class PSDMSE(nn.Module):
    """Power Spectral Density MSE loss with NaN handling."""

    def __init__(self, high_pass: float, low_pass: float) -> None:
        """Create helpers for PSD computation and MSE measurement."""
        super().__init__()
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.psd = NormPSD(self.high_pass, self.low_pass)
        self.mse = nn.MSELoss(reduction='mean')
        print("Using PSD MSE loss for training.")

    def forward(self, preds: Tensor, labels: Tensor, fs: int) -> Tensor:
        """Compute MSE between normalized PSDs for predictions and labels."""
        # Handle NaN input
        if torch.isnan(preds).any() or torch.isnan(labels).any():
            print("Warning: NaN input to PSD_MSE loss")
            # Replace NaNs with zeros
            preds = torch.nan_to_num(preds, nan=0.0)
            labels = torch.nan_to_num(labels, nan=0.0)
        
        # Calculate normalized PSDs
        pred_psd_norm = self.psd(preds, fs)
        label_psd_norm = self.psd(labels, fs)
        
        # Calculate MSE loss
        loss = self.mse(pred_psd_norm, label_psd_norm)
        
        # Handle NaN loss
        if torch.isnan(loss):
            print("Warning: NaN loss in PSD_MSE")
            return torch.tensor(0.1, device=preds.device, requires_grad=True)
        
        return loss
