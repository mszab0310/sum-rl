from trafficEnv import TrafficEnv
import resources.config as conf

sumo_cmd = "sumo"

cfg = [sumo_cmd, "-c", conf.sumo_cfg, "-r", conf.trips_file]

env = TrafficEnv(cfg, "J25", 3600)

eps = 10

for episode in range(eps):
    done = False
    obs = env.reset()
    d = False
    while not d:
        random_action = env.action_space.sample()
        obs, reward, done, info = env.step(random_action)
        d = done
        print(reward)
