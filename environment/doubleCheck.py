from trafficEnv import TrafficEnv

base_dir = "D:/SUMOS/SimpleJunction/"
sumo_cfg = base_dir + "simpleJunction.sumocfg"
trips_file = base_dir + "trips.trips.xml"
additional_files = base_dir + "simpleJunction.add.xml"
sumo_cmd = "sumo"

cfg = [sumo_cmd, "-c", sumo_cfg, "-r", trips_file]

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
        # print(reward)
