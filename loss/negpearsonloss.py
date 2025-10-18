from __future__ import print_function, division
import torch


import numpy as np
from torchvision import transforms
from torch import nn


class Neg_Pearson(nn.Module):
    """
    Neg_Pearson loss for rPPG signal reconstruction with NaN handling.
    """
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        
    def forward(self, preds, labels):
        loss = 0
        for i in range(preds.shape[0]):
            # Check for invalid data
            if torch.isnan(preds[i]).any() or torch.isnan(labels[i]).any():
                print("Warning: NaN values detected in inputs to Neg_Pearson loss")
                continue
                
            sum_x = torch.sum(preds[i])
            sum_y = torch.sum(labels[i])
            sum_xy = torch.sum(preds[i]*labels[i])
            sum_x2 = torch.sum(torch.pow(preds[i], 2))
            sum_y2 = torch.sum(torch.pow(labels[i], 2))
            N = preds.shape[1]
            
            # Add epsilon to avoid division by zero
            eps = 1e-8
            numerator = (N*sum_xy - sum_x*sum_y)
            denominator = torch.sqrt((N*sum_x2 - torch.pow(sum_x,2) + eps) * 
                                  (N*sum_y2 - torch.pow(sum_y,2) + eps))
            
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
            
        loss = loss/preds.shape[0]
        
        # Final safety check
        if torch.isnan(loss):
            print("Warning: NaN loss produced in Neg_Pearson")
            return torch.tensor(0.1, device=preds.device, requires_grad=True)
            
        return loss
