# docs: https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from models.cnnmodel import CNN_RL
import envs.register_env  


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "JammerLocalization-v0"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 2000
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 4

    # Logging helper
    smooth_window: int = 50  # moving average window for episodic return


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        import gymnasium as gym
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env
    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.single_observation_space.shape  
        assert len(obs_shape) == 3, f"Expected obs (C,H,W). Got: {obs_shape}"
        c, h, w = obs_shape
        assert (h, w) == (32, 32), f"Expected (H,W)=(32,32). Got: {(h,w)}"

        num_actions = env.single_action_space.n
        self.model = CNN_RL(input_channels=c, num_actions=num_actions)

    def forward(self, x):
        return self.model(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / float(duration)
    return max(slope * float(t) + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "DQN script supports num_envs=1 (standard CleanRL DQN)."

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space supported"

    # Networks
    q_network = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # Replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # For moving average episodic return plot
    return_window = deque(maxlen=args.smooth_window)

    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # Epsilon-greedy
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )
        writer.add_scalar("charts/epsilon", epsilon, global_step)

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                q_values = q_network(torch.tensor(obs, device=device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Episode logging
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_r = float(info["episode"]["r"])
                    ep_l = int(info["episode"]["l"])
                    return_window.append(ep_r)
                    ep_r_smooth = float(np.mean(return_window))

                    writer.add_scalar("charts/episodic_return", ep_r, global_step)
                    writer.add_scalar("charts/episodic_return_smooth", ep_r_smooth, global_step)
                    writer.add_scalar("charts/episodic_length", ep_l, global_step)
                    
                    if "distance_to_target" in info:
                        writer.add_scalar("charts/localization_error", info["distance_to_target"], global_step)
                    elif "error" in info:
                        writer.add_scalar("charts/localization_error", info["error"], global_step)

                    print(f"step={global_step} return={ep_r:.3f} len={ep_l} eps={epsilon:.3f}")

        # Replay buffer add (handle truncation final obs)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Train
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                target_q = target_network(data.next_observations).max(dim=1).values
                td_target = data.rewards.flatten() + args.gamma * target_q * (1.0 - data.dones.flatten())

            current_q = q_network(data.observations).gather(1, data.actions).squeeze(1)
            loss = F.mse_loss(current_q, td_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # TD loss plot (required)
            writer.add_scalar("losses/td_loss", loss.item(), global_step)
            # Optional useful plot
            writer.add_scalar("losses/q_values_mean", current_q.mean().item(), global_step)

        # Target net update
        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            for tp, p in zip(target_network.parameters(), q_network.parameters()):
                tp.data.copy_(args.tau * p.data + (1.0 - args.tau) * tp.data)

        # SPS plot
        if global_step % 1000 == 0 and global_step > 0:
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            
        if global_step % 50000 == 0:
            save_path = f"runs/{run_name}/model_{global_step}.pth"
            torch.save(q_network.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
            
    final_save_path = f"runs/{run_name}/{args.exp_name}_final.pth"
    torch.save(q_network.state_dict(), final_save_path)
    print(f"Training Complete. Final model saved to {final_save_path}")

    envs.close()
    writer.close()
