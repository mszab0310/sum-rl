from environment.sumos.gym_env import TrafficLightEnvironment2
from environment.resources import config

sumo_cmd = "sumo"

cfg = [sumo_cmd, "-c", config.sumo_cfg, "-r", config.trips_file]

env = TrafficLightEnvironment2(cfg, 5, 0, 60, 5, 3600)

eps = 10

for episode in range(eps):
    done = False
    obs = env.reset()
    d = False
    while not d:
        random_action = env.action_space.sample()
        obs, reward, done, info = env.step(random_action)
        d = done
        print("Reward", reward)
