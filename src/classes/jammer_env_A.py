import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import logging
import random
import torch
from PIL import Image

from utils.utilities_tsai import get_dataset
from utils.logger_utils import get_logger

def compute_distance(pos1, pos2, metric="euclidean"):
    """
    Computes distance between two points.
    Expects pos1 and pos2 to be in the same scale (e.g., both raw meters).
    """
    vec = pos2 - pos1
    if metric == "euclidean":
        dist = np.linalg.norm(vec)
    elif metric == "manhattan":
        dist = np.sum(np.abs(vec))
    elif metric == "chebyshev":
        dist = np.max(np.abs(vec))
    else:
        raise ValueError("Unsupported distance metric.")
    return dist

class JammerLocalizationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, dataset, 
                 horiz_step_meters=0.5,  # Step size in REAL METERS (e.g., 0.5m)
                 vert_step_meters=0.5,   # Step size in REAL METERS (e.g., 0.5m)
                 max_steps=300, 
                 xy_tol=2.0,             # Target radius in REAL METERS (Thesis Criteria)
                 z_tol=0.25,             # Target height tol in REAL METERS
                 distance_metric="euclidean", # Euclidean is best for progress rewards
                 render_mode=None):
        
        super(JammerLocalizationEnv, self).__init__()
        
        self.render_mode = render_mode
        self.logger = get_logger("JammerLocalizationEnv", level=logging.WARNING)
        JammerLocalizationEnv._env_counter = getattr(JammerLocalizationEnv, '_env_counter', 0) + 1
        self.env_id = JammerLocalizationEnv._env_counter
        self.logger.info(f"Created JammerLocalizationEnv with ID {self.env_id}")

        self.dataset = dataset
        self.distance_metric = distance_metric

        # --- 1. COORDINATE SETUP ---
        # Raw positions (Meters) - Used for REWARD and SUCCESS calculation
        self.positions_raw = np.array([dataset[i][1].numpy() for i in range(len(dataset))])
        self.pos_min = self.positions_raw.min(axis=0)
        self.pos_max = self.positions_raw.max(axis=0)
        
        # Dimensions for scaling logic (Max coords - Min coords)
        self.space_dimensions = self.pos_max - self.pos_min + 1e-8

        # Normalized positions (0.0 to 1.0) - Used for KDTree (Observation lookup)
        # We keep this normalized so the CNN input remains consistent.
        self.positions_norm = (self.positions_raw - self.pos_min) / self.space_dimensions
        self.kd_tree = KDTree(self.positions_norm)

        # --- 2. RL SPACES ---
        self.action_space = gym.spaces.Discrete(24)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 32, 32), dtype=np.float32)

        # --- 3. CONFIGURATION ---
        self.horiz_step_meters = horiz_step_meters
        self.vert_step_meters = vert_step_meters
        self.max_steps = max_steps
        
        
        # Success Criteria 
        self.xy_tol = xy_tol
        self.z_tol = z_tol

        # Internal State
        self.current_step = 0
        self.agent_position_norm = None 
        self.goal_position_raw = None   
        self.last_dist = None           
        self.trajectory = []

    def _get_nearest_signal_iq(self):
        # Query KDTree using NORMALIZED coordinates
        _, indices = self.kd_tree.query(self.agent_position_norm.reshape(1, -1), k=1)
        nearest_index = indices[0, 0]
        sample = self.dataset[nearest_index]
        obs = sample[0]  # (8, 32, 32)
        return obs

    def _get_observation(self):
        return self._get_nearest_signal_iq()

    def _to_meters(self, pos_norm):
        """Helper: Converts normalized (0-1) position to real meters."""
        return pos_norm * self.space_dimensions + self.pos_min

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 1. Randomize Agent Position
        while True:
            idx = random.randint(0, len(self.positions_norm) - 1)
            self.agent_position_norm = self.positions_norm[idx].copy()
            break
        
        self.trajectory = [self.agent_position_norm.copy()]

        # 2. Randomize Goal Position 
        while True:
            goal_idx = random.randint(0, len(self.positions_raw) - 1)
            candidate_raw = self.positions_raw[goal_idx]
            candidate_norm = self.positions_norm[goal_idx]

            # Ensure goal is not on top of agent 
            agent_raw = self._to_meters(self.agent_position_norm)
            if np.linalg.norm(candidate_raw - agent_raw) >= 2.0: 
                self.goal_position_raw = candidate_raw.copy()
                break
        
    
        current_pos_raw = self._to_meters(self.agent_position_norm)
        self.last_dist = compute_distance(current_pos_raw, self.goal_position_raw, self.distance_metric)

        self.current_step = 0
        self.total_episode_reward = 0.0
        self.episode_length = 0

        obs = self._get_observation()
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()

        return obs, {}
    
    def step(self, action):
        self.current_step += 1

        # 1. Decode Action to Movement 
        horizontal_idx = action // 3
        vertical_idx = action % 3

        angle_deg = horizontal_idx * 45
        angle_rad = np.deg2rad(angle_deg)
        
        # Calculate delta in REAL METERS
        delta_x_m = self.horiz_step_meters * np.cos(angle_rad)
        delta_y_m = self.horiz_step_meters * np.sin(angle_rad)
        delta_z_m = self.vert_step_meters * (vertical_idx - 1)
        
        delta_meters = np.array([delta_x_m, delta_y_m, delta_z_m])

        # 2. Apply Movement
        delta_norm = delta_meters / self.space_dimensions 
        
        self.agent_position_norm += delta_norm
        self.agent_position_norm = np.clip(self.agent_position_norm, 0.0, 1.0)
        self.trajectory.append(self.agent_position_norm.copy())

        # 3. Calculate Distances 
        current_pos_raw = self._to_meters(self.agent_position_norm)
        current_dist = compute_distance(current_pos_raw, self.goal_position_raw, self.distance_metric)
        
        # Calculate 3D Euclidean error for logging
        euclidean_error = np.linalg.norm(current_pos_raw - self.goal_position_raw)

        # 4. Check Success Conditions
        
        xy_dist = np.linalg.norm(current_pos_raw[:2] - self.goal_position_raw[:2])
        z_dist = np.abs(current_pos_raw[2] - self.goal_position_raw[2])
        
        is_success = (xy_dist < self.xy_tol) and (z_dist < self.z_tol)

        # 5. Calculate Reward (Progress + Step + Terminal)
        
        # A. Progress Reward (Old Dist - New Dist)
        # Positive if getting closer, Negative if moving away
        progress = self.last_dist - current_dist
        self.last_dist = current_dist

        # B. Step Penalty (Encourage speed)
        step_penalty = -0.01

        # C. Terminal Bonus
        terminal_bonus = 20.0 if is_success else 0.0

        reward = progress + step_penalty + terminal_bonus

        self.total_episode_reward += reward
        self.episode_length += 1

        terminated = is_success
        truncated = self.current_step >= self.max_steps

        info = {}
        if terminated or truncated:
            info["episode"] = {
                "r": self.total_episode_reward,
                "l": self.episode_length
            }
            info["distance_to_target"] = euclidean_error
            info["error"] = euclidean_error
            info["is_success"] = is_success
            
            info["xy_error"] = xy_dist
            info["z_error"] = z_dist
            info["agent_position"] = current_pos_raw
            info["target_position"] = self.goal_position_raw


        obs = self._get_observation()
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()

        return obs, reward, terminated, truncated, info
    
    def render(self, mode="rgb_array"):
        fig = plt.figure(figsize=(6, 6)) 
        ax = fig.add_subplot(111, projection='3d')
        

        ax.scatter(self.positions_raw[:, 0], self.positions_raw[:, 1], self.positions_raw[:, 2], 
                   c='green', alpha=0.05, s=5, label="Valid Nodes")

        if len(self.trajectory) > 1:
            traj_norm = np.array(self.trajectory)
     
            traj_raw = traj_norm * self.space_dimensions + self.pos_min
            
            # Plot Red Line
            ax.plot(traj_raw[:, 0], traj_raw[:, 1], traj_raw[:, 2], 
                    color='red', linewidth=2, label="Path")
            
        # 3. Plot Current Agent Position (Blue)
        agent_raw = self._to_meters(self.agent_position_norm)
        ax.scatter(*agent_raw, color='blue', s=80, label="Agent", edgecolor='white')
        
        # 4. Plot Goal Position (Black X)
        ax.scatter(*self.goal_position_raw, color='black', s=100, label="Goal", marker='X')
        
        # 5. Set Limits to Exact Dataset Boundaries 
        ax.set_xlim(self.pos_min[0], self.pos_max[0])
        ax.set_ylim(self.pos_min[1], self.pos_max[1])
        ax.set_zlim(self.pos_min[2], self.pos_max[2])
        
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Height (m)")

        
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)[..., :3]
        
        plt.close(fig)
        return img_array
    
    def close(self):
        pass