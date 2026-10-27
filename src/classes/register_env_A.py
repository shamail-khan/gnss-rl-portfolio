import gymnasium as gym
from gymnasium.envs.registration import register

from envs.jammer_env_norm import JammerLocalizationEnv
from utils.utilities_tsai import get_dataset

def make_jammer_env(render_mode=None, train=True):
    # Load the dataset (Same as before)
    dataset = get_dataset(
        task_type="comb",
        receiver_type="crpa",
        subtask_types=[5, 6, 7, 8, 9],
        train_types=[1, 1, 1, 1, 1],
        return_x_data=0,
        reshape_crpa=True,
        norm=True,
        image_size=[1, 1024, 34],
        train=train
    )


    env = JammerLocalizationEnv(
        dataset=dataset,
    
        horiz_step_meters=1.0,  
      
        vert_step_meters=0.5,   
        
        max_steps=300,
        
       
        xy_tol=2.0,             
        z_tol=0.25,             
        
        distance_metric="euclidean",  
        render_mode=render_mode,
    )
    return env

register(
    id="JammerLocalization-v0",
    entry_point="envs.register_env:make_jammer_env",
    
)



