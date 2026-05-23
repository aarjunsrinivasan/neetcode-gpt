import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.l1 = nn.Linear(784, 512)
        self.rel1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.2)
        self.l2 = nn.Linear(512, 10)
        self.act2 = nn.Sigmoid()


    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x = self.dp1(self.rel1 (self.l1(images)))
        return self.act2(self.l2(x))
        
