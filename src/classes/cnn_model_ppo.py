import torch
import torch.nn as nn
from torch.distributions import Categorical

class CNN_PPO(nn.Module):
    """
    CNN backbone + actor/critic heads for PPO.

    Expected input: (B, C, H, W) e.g. (B, 4, 32, 32)
    Discrete action space: outputs logits (B, num_actions)
    Value function: outputs value (B,)
    """
    def __init__(self, input_channels: int = 4, num_actions: int = 24, hidden_dim: int = 256):
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)

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

        
        self.shared = nn.Sequential(
            nn.Linear(n_flat, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

       
        self.actor = nn.Linear(self.hidden_dim, self.num_actions)

        self.critic = nn.Linear(self.hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().contiguous()
        z = self.features(x)
        z = z.view(z.size(0), -1)
        z = self.shared(z)
        return z

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        z = self._forward_features(x)
        return self.critic(z).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        """
        Returns: action, logprob, entropy, value
        """
        z = self._forward_features(x)
        logits = self.actor(z)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(z).squeeze(-1)
        return action, logprob, entropy, value

