import os
import time

from stable_baselines3 import PPO

from environment.resources import config
from environment.sumos.gym_env import TrafficLightEnvironment2

sumo_cmd = "sumo"

cfg = [sumo_cmd, "-c", config.sumo_cfg, "-r", config.trips_file]

env = TrafficLightEnvironment2(cfg, 5, 0, 60, 5, 3600)

models_dir = f"models/PPO-{int(time.time())}-today"
logdir = f"logs/PPO-{int(time.time())}-today"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIME_STEPS = 10000
for i in range(1, 1000):
    model.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIME_STEPS * i}")

env.close()
