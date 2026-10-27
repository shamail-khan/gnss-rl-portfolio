import os
import numpy as np
import torch
from gymnasium.envs.registration import register

from utils.utilities_tsai import get_dataset
from envs.jammer_env import JammerLocalizationRings3D

_PRELOAD_CACHE = {}

def _preload_combo_once(load_in_ram=True, debug=False, train=True):
    key = ("default", load_in_ram, train)
    if key in _PRELOAD_CACHE:
        return _PRELOAD_CACHE[key]

    ds = get_dataset(
        task_type="comb",
        receiver_type="crpa",
        subtask_types=[5, 6, 7, 8, 9],
        train_types=[1, 1, 1, 1, 1],
        norm=True,
        train=train,            
        return_x_data=1,         
        reshape_crpa=False,      
        load_in_ram=load_in_ram, 
    )   
    N = len(ds)

    x0, *_ = ds[0]
    if not torch.is_tensor(x0):
        x0 = torch.as_tensor(x0)

    assert x0.ndim == 2, f"Expected (C,L) like (8,1024). Got {tuple(x0.shape)}"
    C, L = x0.shape
    assert C == 8, f"Expected C=8. Got {C}"
    assert L == 1024, f"Expected L=1024. Got {L}"

    obs_np = np.empty((N, C, L), dtype=np.float32)

    for i in range(N):
        x, *_ = ds[i]  
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        obs_np[i] = x.float().cpu().numpy()


    xyz = ds.labels[0]
    if torch.is_tensor(xyz):
        xyz = xyz.cpu().numpy()
        
    pos_np = xyz[:, :3].astype(np.float32)
    
    obs_np = np.ascontiguousarray(obs_np)
    pos_np = np.ascontiguousarray(pos_np)

    _PRELOAD_CACHE[key] = (obs_np, pos_np)
    return obs_np, pos_np


def make_jammer_env_rings3d(render_mode=None, debug=False, train=True, **env_kwargs):
    

    obs_np, pos_np = _preload_combo_once(load_in_ram=True, debug=debug, train=train)

    defaults = dict(
        n_floors=4,
        n_grids=24,
        n_rings=5,
        max_steps=300,
        normalize_positions=False,
        progress_scale=5.0,
        success_bonus=20.0,
        step_cost=0.01,
        xy_tol=2.0,
        z_tol=0.25,
        doorway_deg_tol=30.0,
        ring_outside_index=0,
        doorway_on_all_rings=True,
        debug=debug,
        render_mode=render_mode,
    )
    defaults.update(env_kwargs)

    return JammerLocalizationRings3D(
        obs_array=obs_np,
        pos_array=pos_np,
        **defaults,
    )


register(
    id="JammerLocalizationRings3D-v0",
    entry_point="envs.register_env:make_jammer_env_rings3d",
)









