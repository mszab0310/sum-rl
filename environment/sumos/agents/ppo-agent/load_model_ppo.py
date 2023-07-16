import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from environment.resources import config
from environment.sumos.gym_env import TrafficLightEnvironment2


def draw_states_steps(states):
    time_intervals = list(range(len(states)))
    plt.step(time_intervals, states, where='post')

    plt.xlabel('Time')
    plt.ylabel('Light State')
    plt.title('Green and Yellow Light Phases')
    plt.yticks([-1, 1], ['Yellow', 'Green'])
    plt.show()


def draw_avg(max_times):
    print("AVG", sum(max_times) / len(max_times))
    plt.plot(max_times)
    plt.ylabel("Max waiting times")
    plt.xlabel("value count")
    plt.show()


models_dir = "models/PPO-1687178824"
model_path = f"{models_dir}/80000.zip"

sumo_cmd = "sumo-gui"  # change this to sumo if gui is not needed

cfg = [sumo_cmd, "-c", config.sumo_cfg, "-r", config.trips_file]

traffic = TrafficLightEnvironment2(cfg, 5, 0, 60, 5, 3600)

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
    states_codes = traffic.get_states_code_array()
    # draw_states_steps(states_codes[:50])
    # only one of these should be uncommented
    draw_avg(max_times)

traffic.close()
