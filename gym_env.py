from stable_baselines3.common.env_checker import check_env

from environment.trafficEnv import TrafficEnv

base_dir = "D:/SUMOS/SimpleJunction/"
sumo_cfg = base_dir + "simpleJunction.sumocfg"
trips_file = base_dir + "trips.trips.xml"
additional_files = base_dir + "simpleJunction.add.xml"
sumo_cmd = "sumo"

cfg = [sumo_cmd, "-c", sumo_cfg, "-r", trips_file]

env = TrafficEnv(cfg, "J25", 3600)

check_env(env)
