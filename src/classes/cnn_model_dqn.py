import torch
import torch.nn as nn


class CNN_RL(nn.Module):
    """
    Simple CNN for DQN on spectrogram inputs.

    Expected input: (B, C, H, W) e.g. (B, 4, 32, 32)
    Output: Q-values (B, num_actions)
    """
    def __init__(self, input_channels: int = 4, num_actions: int = 24):
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_actions = int(num_actions)

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, 32, 32)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_actions),
        )

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        x = x.float().contiguous()
        z = self.features(x)
        z = z.view(z.size(0), -1)
        return self.head(z)
