import io
import math
import random
import time
from typing import Dict, Any, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

# ---------- Fixed z levels ----------
Z_LEVELS = np.array([1.68, 2.18, 2.68, 3.18], dtype=np.float32)  
Z_TOL     = 0.15   
RING_SPACING = 0.30  


def _wrap_angle(theta: np.ndarray) -> np.ndarray:
    out = np.mod(theta, 2.0 * np.pi)
    out[out < 0] += 2.0 * np.pi
    return out

def _ang_diff(a: float, b: float) -> float:
    """smallest absolute difference between two angles in [0, 2π)"""
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)

def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _nearest_idx(val: float, arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    return int(np.argmin(np.abs(arr - val)))

def _as_float32(x):
    return np.asarray(x, dtype=np.float32)


class JammerLocalizationRings3D(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    CW_LARGE, CW_MED, CW_SMALL, \
    CCW_LARGE, CCW_MED, CCW_SMALL, \
    R_OUT, R_IN, UP, DOWN = range(10)


    CW_SKIP = {
        CW_SMALL: 1,
        CW_MED:   3,
        CW_LARGE: 5,
    }
    CCW_SKIP = {
        CCW_SMALL: -1,
        CCW_MED:   -3,
        CCW_LARGE: -5,
    }

    def __init__(
        self,
        obs_array: np.ndarray | None = None,
        pos_array: np.ndarray | None = None,
        dataset: list[tuple[np.ndarray, np.ndarray]] | None = None,
        n_floors: int = 4,
        n_grids: int = 24,
        n_rings: int = 5,
        max_steps: int = 300,

        normalize_positions: bool = False,
        progress_scale: float = 5.0,
        success_bonus: float = 20.0,
        step_cost: float = 0.01,

        xy_tol: float | None = None,
        z_tol: float | None = None,

        doorway_deg_tol: float = 12.0,   
        ring_outside_index: int = 0,     
        doorway_on_all_rings: bool = False,

        debug: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        
        t0 = time.time()
        
        self.dataset = dataset
        self.n_floors = int(n_floors)
        self.n_grids  = int(n_grids)
        self.n_rings  = int(n_rings)
        self.max_steps = int(max_steps)

        self.normalize_positions = bool(normalize_positions)
        self.progress_scale = float(progress_scale)
        self.success_bonus  = float(success_bonus)
        self.step_cost      = float(step_cost)

        self.doorway_rad_tol   = math.radians(float(doorway_deg_tol))
        self.ring_outer_idx    = int(ring_outside_index)
        self.doorway_on_all_rings = bool(doorway_on_all_rings)

        self.debug = bool(debug)
        self.render_mode = render_mode
        
        
        if (obs_array is not None) and (pos_array is not None):
            print(f"[{time.time():.2f}] [DEBUG] JLR3D: Using preloaded arrays.", flush=True)
            self.X = np.asarray(obs_array, dtype=np.float32)
            self.pos_raw = np.asarray(pos_array, dtype=np.float32)
        elif dataset is not None:
            print(f"[{time.time():.2f}] [DEBUG] JLR3D: Converting dataset list -> arrays ...", flush=True)
            self.X = np.asarray([_as_float32(x) for x, _ in dataset], dtype=np.float32)
            self.pos_raw = np.asarray([_as_float32(p) for _, p in dataset], dtype=np.float32)
        else:
            raise ValueError("Provide either (obs_array,pos_array) OR dataset")
        
        assert self.X.ndim in (3,4), f"obs_array must be (N,C,T) or (N,C,H,W); got {self.X.shape}"
        assert self.pos_raw.shape[1] == 3, f"pos must be (N,3); got {self.pos_raw.shape}"
        self.N = int(self.X.shape[0])
        
        print(f"[{time.time():.2f}] [DEBUG] JLR3D: Data ready in {time.time() - t0:.2f}s", flush=True) 

        # Distance-space positions 
        pmin, pmax = self.pos_raw.min(axis=0), self.pos_raw.max(axis=0)
        self.pos_norm = (self.pos_raw - pmin) / (pmax - pmin + 1e-8)
        self.pos_for_dist = self.pos_norm if self.normalize_positions else self.pos_raw

        # Success tolerances
        if xy_tol is None:
            self.xy_tol = 1.2 if self.normalize_positions else 1.2
        else:
            self.xy_tol = float(xy_tol)
        if z_tol is None:
            self.z_tol  = 0.25 if self.normalize_positions else 0.25
        else:
            self.z_tol = float(z_tol)

        # structure
        t_start_structure = time.time()
        print(f"[{t_start_structure:.2f}] [DEBUG] JLR3D: Starting geometric structure build (_build_floors to _build_reverse_lookup).", flush=True)
        self._build_floors()           # fixed Z_LEVELS + tolerance
        t_start_grids = time.time()
        self._build_grids_per_floor()  # KMeans on XY per floor
        t_end_grids = time.time()
        print(f"[{t_end_grids:.2f}] [DEBUG] JLR3D: Grids (KMeans) build complete. Time: {t_end_grids - t_start_grids:.2f} s", flush=True)
        t_start_rings = time.time()
        self._build_rings_per_grid()   # fixed ring spacing from each grid center
        t_end_rings = time.time()
        print(f"[{t_end_rings:.2f}] [DEBUG] JLR3D: Rings assignment complete. Time: {t_end_rings - t_start_rings:.2f} s", flush=True)
        t_start_neigh = time.time()
        self._precompute_neighbors_and_doorways()  # directional neighbors + doorway masks
        self._build_reverse_lookup()   # idx -> (floor, grid, ring, theta_index)
        t_end_neigh = time.time()
        print(f"[{t_end_neigh:.2f}] [DEBUG] JLR3D: Neighbors/Doorways/Lookup complete. Time: {t_end_neigh - t_start_neigh:.2f} s", flush=True)
        
        t_end_init = time.time()
        print(f"[{t_end_init:.2f}] [DEBUG] JLR3D: All structure complete. Total time (structure): {t_end_init - t_start_structure:.2f} s", flush=True)
        # --- NEW: valid indices (assigned to some floor/grid/ring/theta) ---
        self.valid_idxs = np.where(self.idx_to_ftg[:, 0] != -1)[0]
        if self.valid_idxs.size == 0:
            raise RuntimeError("No valid samples assigned to floors/grids/rings. Check Z_LEVELS/Z_TOL or dataset.")

    
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8, 1024), dtype=np.float32)

        self.cur_idx: int | None = None
        self.goal_idx: int | None = None
        self.cur_ftg: Tuple[int, int, int, int] | None = None
        self.goal_pos: np.ndarray | None = None
        self.prev_dist = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.grid_hops_count = 0
        self.trajectory: List[np.ndarray] = []

        if self.debug:
            print(f"[Env] normalize_positions={self.normalize_positions} | xy_tol={self.xy_tol:.4f} | z_tol={self.z_tol:.4f}")
            print(f"[Env] floors={self.n_floors}, grids/floor={self.n_grids}, rings/grid={self.n_rings}")

    def _build_floors(self):
        """Assign floor ids using fixed Z_LEVELS."""
        if self.n_floors != len(Z_LEVELS):
            raise ValueError(f"n_floors={self.n_floors} must match len(Z_LEVELS)={len(Z_LEVELS)}")

        z = self.pos_raw[:, 2]
        self.floor_id = np.full(len(z), -1, dtype=int)
        for k, z_star in enumerate(Z_LEVELS):
            mask = np.abs(z - z_star) <= Z_TOL
            self.floor_id[mask] = k

        if np.any(self.floor_id == -1):
            # We ignore the diagonals (moving between floors) — they’ll just be unassigned.
            if self.debug:
                n_un = int(np.sum(self.floor_id == -1))
                print(f"[Env] warning: {n_un} samples not assigned to any Z floor (outside ±{Z_TOL}m).")

        self.floor_z_center = Z_LEVELS.copy()

    def _build_grids_per_floor(self):
        """KMeans on (x,y) within each floor to n_grids -> store centers & assignments."""
        N = len(self.pos_raw)
        self.grid_id = np.full(N, -1, dtype=int)
        self.grid_centers: List[np.ndarray] = []              # [floor] -> (n_grids, 2)
        self.nodes_by_floor_grid: List[List[List[int]]] = []  # [floor][grid] -> list of dataset indices

        for k in range(self.n_floors):
            idxk = np.where(self.floor_id == k)[0]
            xy = self.pos_raw[idxk, :2]

            if len(xy) == 0:
                centers = np.zeros((self.n_grids, 2), dtype=np.float32)
                self.grid_centers.append(centers)
                self.nodes_by_floor_grid.append([[] for _ in range(self.n_grids)])
                continue

            n_clusters = min(self.n_grids, len(xy))
            km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
            gid_raw = km.fit_predict(xy)
            centers_k = km.cluster_centers_.astype(np.float32)

            centers = np.zeros((self.n_grids, 2), dtype=np.float32)
            centers[:n_clusters] = centers_k
            self.grid_centers.append(centers)

            nodes_fg = [[] for _ in range(self.n_grids)]
            for g in range(n_clusters):
                members = idxk[np.where(gid_raw == g)[0]].tolist()
                nodes_fg[g] = members
                self.grid_id[idxk[gid_raw == g]] = g
            self.nodes_by_floor_grid.append(nodes_fg)

    def _build_rings_per_grid(self):
        """Assign ring ids by fixed radial bands from grid center. j=0 is outermost."""
        if self.n_rings <= 0:
            raise ValueError("n_rings must be >= 1")

        self.rings: List[List[List[Dict[str, np.ndarray]]]] = []  
        self.r_mean: List[List[List[float]]] = []
        self.theta_list: List[List[List[np.ndarray]]] = []

        for k in range(self.n_floors):
            rings_k, rmean_k, thetas_k = [], [], []
            centers = self.grid_centers[k]

            for g in range(self.n_grids):
                ids = self.nodes_by_floor_grid[k][g]

                if len(ids) == 0:
                    empty_ring = dict(idx=np.array([], dtype=int),
                                      theta=np.array([], dtype=float),
                                      r=np.array([], dtype=float))
                    rings_k.append([empty_ring for _ in range(self.n_rings)])
                    rmean_k.append([0.0 for _ in range(self.n_rings)])
                    thetas_k.append([np.array([]) for _ in range(self.n_rings)])
                    continue

                xy = self.pos_raw[ids, :2]
                cxy = centers[g]
                rel = xy - cxy
                r = np.linalg.norm(rel, axis=1)                       
                theta = _wrap_angle(np.arctan2(rel[:, 1], rel[:, 0]))

                r_band = np.floor(r / RING_SPACING).astype(int)
                unique_bands = np.unique(r_band)
                band_order = np.argsort(-unique_bands)
                band_to_ring = {unique_bands[b]: j for j, b in enumerate(band_order)}
                lab = np.array([min(band_to_ring[b], self.n_rings - 1) for b in r_band], dtype=int)

                rings_g, rmean_g, thetas_g = [], [], []
                for j in range(self.n_rings):
                    mask = (lab == j)
                    ids_j = np.array(ids, dtype=int)[mask]
                    r_j = r[mask]
                    th_j = theta[mask]
                    if ids_j.size > 0:
                        order_th = np.argsort(th_j)
                        ids_j = ids_j[order_th]
                        r_j = r_j[order_th]
                        th_j = th_j[order_th]
                        rmed = float(np.median(r_j))
                    else:
                        rmed = 0.0

                    rings_g.append(dict(idx=ids_j, theta=th_j, r=r_j))
                    rmean_g.append(rmed)
                    thetas_g.append(th_j)

                rings_k.append(rings_g)
                rmean_k.append(rmean_g)
                thetas_k.append(thetas_g)

            self.rings.append(rings_k)
            self.r_mean.append(rmean_k)
            self.theta_list.append(thetas_k)

    def _precompute_neighbors_and_doorways(self):
        """
        Per floor:
          - directional neighbor for each grid (N/E/S/W) by angle-to-target heuristic
          - doorway angles per grid (N/E/S/W)
          - per-ring doorway masks (θ within ±doorway tol)
        """
        self.neigh = []       
        self.door_theta = []  
        self.door_mask = []   

        for k in range(self.n_floors):
            centers = self.grid_centers[k]   
            G = centers.shape[0]

            neigh_k = {d: np.arange(G) for d in ['N', 'E', 'S', 'W']}
            for g in range(G):
                v = centers - centers[g]
                d = np.linalg.norm(v, axis=1)
                ok = np.where(d > 1e-6)[0]
                if ok.size == 0:
                    continue
                ang = np.arctan2(v[ok, 1], v[ok, 0])
                target = {'E': 0.0, 'N': np.pi/2, 'W': np.pi, 'S': -np.pi/2}
                for name, a0 in target.items():
                    d_ang = np.abs(np.angle(np.exp(1j*(ang - a0))))
                    best_local = np.lexsort((d[ok], d_ang))[0]
                    neigh_k[name][g] = ok[best_local]
            self.neigh.append(neigh_k)
            door_theta_k: List[Dict[str, float]] = []
            door_mask_k:  List[List[Dict[str, np.ndarray]]] = []

            for g in range(G):
                doors_g: Dict[str, float] = {}
                for name in ['N','E','S','W']:
                    g2 = neigh_k[name][g]
                    v = centers[g2] - centers[g]
                    th = math.atan2(v[1], v[0])
                    doors_g[name] = _wrap_angle(np.array([th]))[0]
                door_theta_k.append(doors_g)

                masks_per_ring: List[Dict[str, np.ndarray]] = []
                for j in range(self.n_rings):
                    th_ring = self.rings[k][g][j]['theta']
                    ring_masks = {}
                    if th_ring.size == 0:
                        for name in ['N','E','S','W']:
                            ring_masks[name] = np.zeros(0, dtype=bool)
                    else:
                        for name in ['N','E','S','W']:
                            th_star = doors_g[name]
                            ring_masks[name] = np.array(
                                [_ang_diff(t, th_star) <= self.doorway_rad_tol for t in th_ring],
                                dtype=bool
                            )
                    masks_per_ring.append(ring_masks)
                door_mask_k.append(masks_per_ring)

            self.door_theta.append(door_theta_k)
            self.door_mask.append(door_mask_k)

    def _build_reverse_lookup(self):
        """O(1) mapping: dataset index -> (floor, grid, ring, theta_index)"""
        N = self.N
        self.idx_to_ftg = np.full((N, 4), -1, dtype=np.int32)
        for k in range(self.n_floors):
            for g in range(self.n_grids):
                for j in range(self.n_rings):
                    idxs = self.rings[k][g][j]['idx']
                    for i_th, idx in enumerate(idxs):
                        self.idx_to_ftg[int(idx)] = [k, g, j, i_th]

    def _decode_idx(self, idx: int) -> Tuple[int,int,int,int]:
        ftg = self.idx_to_ftg[idx]
        if ftg[0] == -1:
            raise ValueError(f"Dataset index {idx} not found in structure; check geometry/tolerances.")
        return int(ftg[0]), int(ftg[1]), int(ftg[2]), int(ftg[3])

    def _node_pos_for_dist(self, idx: int) -> np.ndarray:
        return self.pos_for_dist[idx]

    def _obs_from_idx(self, idx: int) -> np.ndarray:
        return self.X[idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- NEW: sample only from valid_idxs to avoid infinite loops ---
        self.cur_idx = int(self.np_random.choice(self.valid_idxs))

        p0 = self._node_pos_for_dist(self.cur_idx)
        min_xy = (0.10 if self.normalize_positions else 1.0)

        # sample goal until far enough in XY
        max_tries = 2000
        for _ in range(max_tries):
            goal = int(self.np_random.choice(self.valid_idxs))
            p1 = self._node_pos_for_dist(goal)
            if np.linalg.norm(p0[:2] - p1[:2]) >= min_xy:
                self.goal_idx = goal
                break
        else:
    # fallback: accept any goal to prevent hanging forever
            self.goal_idx = int(self.np_random.choice(self.valid_idxs))


        self.cur_ftg = self._decode_idx(self.cur_idx)
        self.goal_pos = self._node_pos_for_dist(self.goal_idx)
        self.prev_dist = _euclid(self._node_pos_for_dist(self.cur_idx), self.goal_pos)
        self.total_reward = 0.0
        self.steps = 0
        self.grid_hops_count = 0
        self.trajectory = [self.pos_raw[self.cur_idx].copy()]

        return self._obs_from_idx(self.cur_idx), {}

    def step(self, action: int):
        self.steps += 1
        k, g, j, i_th = self.cur_ftg
        ring = self.rings[k][g][j]
        ids_j = ring['idx']
        n_ang = int(ids_j.size)

        def snap_angle(th_target: float, k2: int, g2: int, j2: int) -> int:
            th_arr = self.rings[k2][g2][j2]['theta']
            return _nearest_idx(th_target, th_arr)

        th_now = float(ring['theta'][i_th % n_ang]) if n_ang > 0 else 0.0

        next_ftg = [k, g, j, i_th]
        did_grid_hop = False

        if action in self.CW_SKIP:
            if n_ang > 0:
                skip = self.CW_SKIP[action]
                next_ftg[3] = (i_th + skip) % n_ang

        elif action in self.CCW_SKIP:
            if n_ang > 0:
                skip = self.CCW_SKIP[action]
                next_ftg[3] = (i_th + skip) % n_ang

        elif action == self.R_OUT:
            allowed_door = self.doorway_on_all_rings or (j == self.ring_outer_idx)
            if allowed_door and n_ang > 0:
                masks = self.door_mask[k][g][j]
                door_hits = []
                for name in ['N','E','S','W']:
                    mask = masks[name]
                    if mask.size and mask[i_th % n_ang]:
                        th_star = self.door_theta[k][g][name]
                        door_hits.append((name, _ang_diff(th_now, th_star)))
                if door_hits:
                    door_hits.sort(key=lambda t: t[1])
                    name = door_hits[0][0]
                    g2 = int(self.neigh[k][name][g])
                    i2 = snap_angle(th_now, k, g2, j) 
                    next_ftg = [k, g2, j, i2]
                    did_grid_hop = True
                else:
                    j2 = max(0, j - 1)
                    i2 = snap_angle(th_now, k, g, j2)
                    next_ftg = [k, g, j2, i2]
            else:
                j2 = max(0, j - 1)
                i2 = snap_angle(th_now, k, g, j2)
                next_ftg = [k, g, j2, i2]

        elif action == self.R_IN:
            j2 = min(self.n_rings - 1, j + 1)
            i2 = snap_angle(th_now, k, g, j2)
            next_ftg = [k, g, j2, i2]

        elif action == self.UP:
            k2 = min(self.n_floors - 1, k + 1)
            c_src = self.grid_centers[k][g]
            centers_tgt = self.grid_centers[k2]
            d2 = np.linalg.norm(centers_tgt - c_src[None, :], axis=1)
            g2 = int(np.argmin(d2))
            i2 = snap_angle(th_now, k2, g2, j)
            next_ftg = [k2, g2, j, i2]

        elif action == self.DOWN:
            k2 = max(0, k - 1)
            c_src = self.grid_centers[k][g]
            centers_tgt = self.grid_centers[k2]
            d2 = np.linalg.norm(centers_tgt - c_src[None, :], axis=1)
            g2 = int(np.argmin(d2))
            i2 = snap_angle(th_now, k2, g2, j)
            next_ftg = [k2, g2, j, i2]

        else:
            raise ValueError(f"Invalid action id {action} (expected 0..9)")

        k2, g2, j2, i2 = next_ftg
        ring2 = self.rings[k2][g2][j2]
        if ring2['idx'].size == 0:
            next_idx = self.cur_idx
            next_ftg = [k, g, j, i_th] 
        else:
            next_idx = int(ring2['idx'][i2 % ring2['idx'].size])

        if did_grid_hop:
            self.grid_hops_count += 1

        nxt_pos = self._node_pos_for_dist(next_idx)
        d_prev = self.prev_dist
        d_cur  = _euclid(nxt_pos, self.goal_pos)
        progress = (d_prev - d_cur)
        reward = self.progress_scale * progress - self.step_cost

        xy_err = float(np.linalg.norm(nxt_pos[:2] - self.goal_pos[:2]))
        z_err  = abs(float(nxt_pos[2] - self.goal_pos[2]))
        success = (xy_err <= self.xy_tol) and (z_err <= self.z_tol)
        if success:
            reward += self.success_bonus

        self.prev_dist = d_cur
        self.cur_idx = next_idx
        self.cur_ftg = (k2, g2, j2, i2)
        self.total_reward += reward
        self.trajectory.append(self.pos_raw[self.cur_idx].copy())

        terminated = success
        truncated = self.steps >= self.max_steps

        info = {}
        if terminated or truncated:
            info["episode"] = {"r": float(self.total_reward), "l": int(self.steps)}
            
            info["distance_to_target"] = float(d_cur)
            info["error"] = float(d_cur)
            info["final_distance"] = float(d_cur)
            info["success"] = bool(terminated)
            info["grid_hops"] = int(self.grid_hops_count)

        return self._obs_from_idx(self.cur_idx), float(reward), terminated, truncated, info

    def render(self, mode="rgb_array"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        k, g, j, i = self.cur_ftg
        z_here = self.pos_raw[self.cur_idx, 2]
        centers = self.grid_centers[k]
        ax.scatter(centers[:,0], centers[:,1], np.full(centers.shape[0], z_here),
                   s=8, c='gray', alpha=0.4, label="Grid Centers")

        cur_raw  = self.pos_raw[self.cur_idx]
        goal_raw = self.pos_raw[self.goal_idx]
        ax.scatter(*cur_raw,  s=60, c='blue', label="Agent")
        ax.scatter(*goal_raw, s=80, marker="X", c='red', label="Goal")

        traj = np.array(self.trajectory)
        if len(traj) > 1:
            ax.plot(traj[:,0], traj[:,1], traj[:,2], c='orange', linestyle=':', alpha=0.7, label="Path")

        x_min, x_max = self.pos_raw[:,0].min(), self.pos_raw[:,0].max()
        y_min, y_max = self.pos_raw[:,1].min(), self.pos_raw[:,1].max()
        z_min, z_max = self.pos_raw[:,2].min(), self.pos_raw[:,2].max()
    
    # 2. Set the axis limits explicitly
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
    
        span_x = x_max - x_min
        span_y = y_max - y_min
        span_z = z_max - z_min
    
    # Note: Since the Z range is tiny (~1.5m) compared to X (30m), 
    # a purely realistic aspect ratio might make the Z-axis look flat. 
    # We multiply span_z by 5 or 10 just to make it visible, while keeping X/Y correct.
        ax.set_box_aspect((span_x, span_y, span_z * 5))
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.legend(loc="upper left")

        canvas = FigureCanvas(fig); canvas.draw()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120); buf.seek(0)
        arr = np.array(Image.open(buf))[..., :3]
        plt.close(fig)
        return arr

    def close(self):
        pass







