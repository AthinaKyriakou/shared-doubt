import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sparsemax import Sparsemax

class SoftmaxLayerProbe(nn.Module):

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        # self.w = nn.Parameter(torch.randn(num_layers)) # → (num_layers, )
        self.w = nn.Parameter(torch.zeros(num_layers))
        self.v = nn.Parameter(torch.randn(hidden_dim)) # → (hidden_dim, )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X.shape(): (batch_size, num_layers, hidden_dim)
        X = X.permute(0, 2, 1) # → (batch_size, hidden_dim, num_layers)
        h = torch.matmul(X, F.softmax(self.w, dim=-1)) # → (batch_size, hidden_dim)
        logits = torch.matmul(h, self.v) + self.bias
        return logits

class UniformLayerProbe(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        self.register_buffer("w", torch.ones(num_layers) / num_layers)  # fixed, uniform
        self.v = nn.Parameter(torch.randn(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch_size, num_layers, hidden_dim)
        X = X.permute(0, 2, 1)  # → (batch_size, hidden_dim, num_layers)
        h = torch.matmul(X, self.w) # → (batch_size, hidden_dim)
        logits = torch.matmul(h, self.v) + self.bias
        return logits

class SparsemaxLayerProbe(nn.Module):

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        # self.w = nn.Parameter(torch.randn(num_layers)) # → (num_layers, )
        self.w = nn.Parameter(torch.zeros(num_layers)) # → (num_layers, )
        self.v = nn.Parameter(torch.randn(hidden_dim)) # → (hidden_dim, )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X.shape(): (batch_size, num_layers, hidden_dim)
        X = X.permute(0, 2, 1) # → (batch_size, hidden_dim, num_layers)
        sparsemax = Sparsemax(dim=-1)
        h = torch.matmul(X, sparsemax(self.w)) # → (batch_size, hidden_dim)
        logits = torch.matmul(h, self.v) + self.bias
        return logits    


class LayerProbe(nn.Module):

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(num_layers)) # → (num_layers, )
        self.v = nn.Parameter(torch.randn(hidden_dim)) # → (hidden_dim, )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X.shape(): (batch_size, num_layers, hidden_dim)
        X = X.permute(0, 2, 1) # → (batch_size, hidden_dim, num_layers)
        h = torch.matmul(X, self.w) # → (batch_size, hidden_dim)
        logits = torch.matmul(h, self.v) + self.bias
        return logits
        

class NeuronProbe(nn.Module):
    """
    A PyTorch implementation of LogisticRegression
    applied to flattened (layer × hidden_dim) features.
    """

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.zeros(num_layers * hidden_dim)) # -> (num_layers * hidden_dim, )
        self.bias = nn.Parameter(torch.zeros(1)) # -> scalar
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X.shape(): (batch_size, num_layers * hidden_dim)
        returns logits of shape (batch_size,)
        """
        logits = torch.matmul(X, self.weight) + self.bias  # -> (batch_size, )
        return logits


class ProbeDataset(Dataset):
    def __init__(self, X_list, y_list):
        assert len(X_list) == len(y_list), "Mismatched X and y lengths"
        self.X = [torch.from_numpy(x).float() for x in X_list]
        self.y = torch.tensor(y_list, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]