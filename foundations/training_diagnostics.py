import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        # x = torch.randn(3,32)
        result = []
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                if isinstance(layer,nn.Linear):
                    if x.dim() >= 2:
                        dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    else:
                        dead_frac = round((x <= 0).float().mean().item(), 4)
                    # o = {'mean':round(x.mean().item(),4) ,'std':round(x.std().item(),4),'dead_fraction':round(torch.sum(x<=0).item()/x.numel(),4)}
                    o = {'mean':round(x.mean().item(),4) ,'std':round(x.std().item(),4),'dead_fraction':dead_frac}
 
                    result.append(o)
        return result


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        result = []
        criterion = nn.MSELoss()
        model.zero_grad()
        out = model(x)
        loss = criterion(y,out)
        loss.backward()
        for layer in model:
            if isinstance(layer,nn.Linear):
                grad = layer.weight.grad
                o = {'mean':round(grad.mean().item(),4),
                'std':round(grad.std().item(),4),
                'norm':round(torch.norm(grad).item(),4)}
                result.append(o)
        return result


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for act_stat ,grad_stat in zip(activation_stats,gradient_stats):
            if act_stat['dead_fraction'] >0.5:
                return 'dead_neurons'

            if grad_stat['norm'] >1000 or act_stat['std'] >10:
                        return 'exploding_gradients'
            if grad_stat['norm'] <1e-5 or act_stat['std'] <0.1:
                        return 'vanishing_gradients'
        return 'healthy'
        
