# drqn_iq.py
import os
import sys
import random
from dataclasses import dataclass
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.tensorboard import SummaryWriter
import collections
import tyro

# --- IMPORT YOUR ENV ---
import envs.register_env
import envs.register_env

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = True

    # Algorithm specific arguments
    env_id: str = "JammerLocalizationRings3D-v0"
    total_timesteps: int = 1000000
    learning_rate: float = 2.5e-4
    buffer_size: int = 5000       # Number of EPISODES
    gamma: float = 0.99
    target_network_frequency: int = 1000
    batch_size: int = 32          # Sequences per batch
    seq_len: int = 8              # Sequence length
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.2
    learning_starts: int = 5000
    train_frequency: int = 4

def make_env(env_id, idx, capture_video, run_name, seed=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            
        return env
    return thunk


# --- 1. RECURRENT NETWORK ---
class RecurrentQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape
        c, l = obs_shape
        
        self.feature_dim = 128
        self.lstm_hidden = 512

        self.encoder = nn.Sequential(
            nn.Conv1d(c, 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, self.feature_dim),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        self.fc_q = nn.Linear(self.lstm_hidden, env.action_space.n)

    def forward(self, x, hidden_state=None):
        if x.dim() == 3: x = x.unsqueeze(1) 

        batch_size, seq_len, c, l = x.shape
        x_flat = x.reshape(batch_size * seq_len, c, l)
        
        features = self.encoder(x_flat)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        q_values = self.fc_q(lstm_out)
        
        return q_values, new_hidden

# --- 2. RECURRENT BUFFER ---
class RecurrentBuffer:
    def __init__(self, max_episodes):
        self.buffer = collections.deque(maxlen=max_episodes)
        
    def push(self, episode_data):
        self.buffer.append(episode_data)
        
    def sample(self, batch_size, seq_len):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        
        b_obs, b_next_obs, b_actions, b_rewards, b_dones, b_masks = [], [], [], [], [], []
        
        for idx in indices:
            episode = self.buffer[idx]
            
            if len(episode) > seq_len:
                start = np.random.randint(0, len(episode) - seq_len + 1)
                sample = episode[start : start + seq_len]
            else:
                sample = episode
            
            s_obs = np.array([x[0] for x in sample])
            s_next_obs = np.array([x[1] for x in sample])
            s_act = np.array([x[2] for x in sample])
            s_rew = np.array([x[3] for x in sample])
            s_don = np.array([x[4] for x in sample])
            
            s_mask = np.ones(len(sample), dtype=np.float32)

            if len(sample) < seq_len:
                pad_len = seq_len - len(sample)
                
            
                obs_pad = np.zeros((pad_len,) + s_obs.shape[1:], dtype=s_obs.dtype)
                nobs_pad = np.zeros((pad_len,) + s_next_obs.shape[1:], dtype=s_next_obs.dtype)
                
                s_obs = np.concatenate([s_obs, obs_pad], axis=0)
                s_next_obs = np.concatenate([s_next_obs, nobs_pad], axis=0)
                
                s_act = np.concatenate([s_act, np.zeros(pad_len, dtype=s_act.dtype)], axis=0)
                s_rew = np.concatenate([s_rew, np.zeros(pad_len, dtype=s_rew.dtype)], axis=0)
                s_don = np.concatenate([s_don, np.ones(pad_len, dtype=s_don.dtype)], axis=0)
 
                
                s_mask = np.concatenate([s_mask, np.zeros(pad_len)], axis=0)

            b_obs.append(s_obs)
            b_next_obs.append(s_next_obs)
            b_actions.append(s_act)
            b_rewards.append(s_rew)
            b_dones.append(s_don)
            b_masks.append(s_mask)
            
        return (torch.tensor(np.array(b_obs), dtype=torch.float32),
                torch.tensor(np.array(b_next_obs), dtype=torch.float32),
                torch.tensor(np.array(b_actions), dtype=torch.long),
                torch.tensor(np.array(b_rewards), dtype=torch.float32),
                torch.tensor(np.array(b_dones), dtype=torch.float32),
                torch.tensor(np.array(b_masks), dtype=torch.float32))

# --- 3. TRAINING LOOP ---
if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, config=vars(args), name=run_name, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    env = make_env(args.env_id, 0, args.capture_video, run_name, args.seed)()
    
    q_network = RecurrentQNetwork(env).to(device)
    target_network = RecurrentQNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    buffer = RecurrentBuffer(max_episodes=args.buffer_size)
    
    curr_episode = []
    obs, _ = env.reset()
    hidden = None
    
    start_time = time.time()
    
    for global_step in range(args.total_timesteps):
        slope = (args.end_e - args.start_e) / (args.exploration_fraction * args.total_timesteps)
        epsilon = max(slope * global_step + args.start_e, args.end_e)
        
        if random.random() < epsilon:
            action = env.action_space.sample()
            
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs).unsqueeze(0).float().to(device)
                q_vals, hidden = q_network(obs_t, hidden)
                action = torch.argmax(q_vals[:, -1, :]).item()
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        curr_episode.append((obs, next_obs, action, reward, float(done)))
        obs = next_obs
        
        if done:
            buffer.push(curr_episode)
            curr_episode = []
            if "episode" in info:
                print(f"step={global_step}, return={info['episode']['r']:.2f}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            obs, _ = env.reset()
            hidden = None 

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            if len(buffer.buffer) >= args.batch_size:
                b_obs, b_next_obs, b_act, b_rew, b_done, b_mask = buffer.sample(args.batch_size, args.seq_len)
                
                b_obs, b_next_obs = b_obs.to(device), b_next_obs.to(device)
                b_act, b_rew = b_act.to(device), b_rew.to(device)
                b_done, b_mask = b_done.to(device), b_mask.to(device)
                
                q_values, _ = q_network(b_obs) 
                q_acted = q_values.gather(2, b_act.unsqueeze(-1)).squeeze(-1)
                
                with torch.no_grad():
                    target_q, _ = target_network(b_next_obs)
                    max_target_q = target_q.max(dim=2)[0]
                    targets = b_rew + args.gamma * max_target_q * (1 - b_done)
                
                #  Numerical safety for division
                element_loss = F.mse_loss(q_acted, targets, reduction='none')
                masked_loss = (element_loss * b_mask).sum() / (b_mask.sum() + 1e-8)
                
                optimizer.zero_grad()
                masked_loss.backward()
                optimizer.step()
                
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", masked_loss.item(), global_step)

                if global_step % args.target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())

    if args.save_model:
        model_path = f"models/{run_name}_drqn.pt"
        os.makedirs("models", exist_ok=True)
        torch.save(q_network.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")

    env.close()
    writer.close()