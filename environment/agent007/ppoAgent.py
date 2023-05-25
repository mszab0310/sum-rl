import os
import time

from stable_baselines3 import PPO

from environment.trafficEnv import TrafficEnv

base_dir = "D:/SUMOS/SimpleJunction/"
sumo_cfg = base_dir + "simpleJunction.sumocfg"
trips_file = base_dir + "trips.trips.xml"
additional_files = base_dir + "simpleJunction.add.xml"
sumo_cmd = "sumo"

cfg = [sumo_cmd, "-c", sumo_cfg, "-r", trips_file]

env = TrafficEnv(cfg, "J25", 3600)

models_dir = f"models/PPO-{int(time.time())}"
logdir = f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = TrafficEnv(cfg, "J25", 3600)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIME_STEPS = 10000
for i in range(1, 100000):
    model.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIME_STEPS}")

env.close()
