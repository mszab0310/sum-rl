import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from environment.sumos.gym_env import TrafficLightEnvironment2
from environment.resources import config


models_dir = "models/A2c-1688462817"
model_path = f"{models_dir}/70000.zip"

sumo_cmd = "sumo-gui"

cfg = [sumo_cmd, "-c", config.sumo_cfg, "-r", config.trips_file]

traffic = TrafficLightEnvironment2(cfg, 5, 0, 60, 5, 3600)

# Define custom objects
custom_objects = {"lr_schedule": lambda x: x, "clip_range": lambda x: x}

model = PPO.load(model_path, env=traffic, custom_objects=custom_objects)

episodes = 1000

for ep in range(episodes):
    obs = traffic.reset()
    done = False
    while not done:
        traffic.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = traffic.step(action)
    max_times = traffic.get_waiting_times_data()
    print("AVG", sum(max_times) / len(max_times))
    plt.plot(max_times)
    plt.ylabel("Max waiting times")
    plt.xlabel("value count")
    plt.show()

traffic.close()
