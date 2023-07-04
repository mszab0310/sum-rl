from stable_baselines3.common.env_checker import check_env

from environment.trafficEnv import TrafficEnv
from environment.resources import config

sumo_cmd = "sumo"

cfg = [sumo_cmd, "-c", config.sumo_cfg, "-r", config.trips_file]


env = TrafficEnv(cfg, "J25", 3600)

check_env(env)
