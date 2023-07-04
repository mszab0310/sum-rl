from stable_baselines3.common.env_checker import check_env
from environment.resources import config


from environment.sumos.gym_env import TrafficLightEnvironment2

sumo_cmd = "sumo"

cfg = [sumo_cmd, "-c", config.sumo_cfg, "-r", config.trips_file]

check_env(TrafficLightEnvironment2(cfg, 5, 0, 60, 5, 3600))
