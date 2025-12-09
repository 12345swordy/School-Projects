import torch
import torch.nn as nn


# ===== Base class with save/load helpers =====
class BaseNet(nn.Module):
    """Shared weight init and PyTorch-native save/load methods."""
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def save(self, path):
        """Save this model’s parameters (state_dict) to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        """Load parameters into this model from disk."""
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)


# ===== Model definitions =====
class SmallNet(BaseNet):
    """Small: hidden layer (64)"""
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)
        self.net = nn.Sequential(
            nn.Linear(self.in_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_size),
        )
        self._init_weights()

    def forward(self, x):
        return self.net(x)


class MediumNet(BaseNet):
    """Medium: hidden layers (128, 64)"""
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)
        self.net = nn.Sequential(
            nn.Linear(self.in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_size),
        )
        self._init_weights()

    def forward(self, x):
        return self.net(x)


class LargeNet(BaseNet):
    """Large: hidden layers (128, 128, 64, 32)"""
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)
        self.net = nn.Sequential(
            nn.Linear(self.in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.out_size),
        )
        self._init_weights()

    def forward(self, x):
        return self.net(x)
