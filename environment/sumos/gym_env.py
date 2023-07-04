import gym
import numpy as np
import traci
from gym import spaces

from environment.sumos.traffic_light import TrafficLight


class TrafficLightEnvironment2(gym.Env):
    def render(self, mode="human"):
        pass

    def __init__(self, sumo_cfg, action_frequency, start_time, max_green, min_green, sim_duration):
        super(TrafficLightEnvironment2, self).__init__()
        self.sumo_cfg = sumo_cfg
        self.action_frequency = action_frequency
        self.start_time = start_time
        self.max_green = max_green
        self.min_green = min_green
        self.sim_duration = sim_duration

        self.done = None
        self.info = {}
        self.observation = None
        self.reward = None
        self.action_space = spaces.Discrete(4)  # 4 green phase for all edges
        # low and high values for each value from the observation space 18 values in total, element 0 indicates the
        # decimal value of one hot encoding of current traffic light state next 8 values are the % of occupancy of
        # each lane (how much % is occupied by cars) next 8 values are the number of cars on each lane with speed <=
        # 0.1/ms - meaning a full stop or waiting last value is a boolean flag encoded in binary that indicates that
        # next 8 values represent the avg waiting time in the current step in that lane
        # if minimum_green_time of time has passed or not since last green phase if the agent gives green to an edge,
        # it should hold it for minimum_green_time seconds for it to be good - it makes it not switch green to red
        # every 2 seconds
        low = np.array(
            [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0,
             0])
        high = np.array(
            [8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100, 100, 100, 100, 100,
             100, 100, 100, 1])
        self.observation_space = spaces.Box(low=low, high=high, shape=(26,), dtype=np.float64)

        self.traffic_env = None
        self.tls_id = None
        self.sumo = None
        self.started = False
        self.ep = 0
        self.green_states = ['GGGGGrrrrrrrrrrr', 'rrrrGGGGGrrrrrrr', 'rrrrrrrrGGGGGrrr', 'GrrrrrrrrrrrGGGG']
        self.yellow_states = ['yyyyyrrrrrrrrrrr', 'rrrryyyyyrrrrrrr', 'rrrrrrrryyyyyrrr', 'yrrrrrrrrrrryyyy']

    def start_sim(self):
        traci.start(self.sumo_cfg, label='t' + str(self.ep))
        self.started = True
        # TODO: obtain programmatically
        self.tls_id = "J25"
        self.sumo = traci
        self.traffic_env = TrafficLight(self.sumo, self, self.action_frequency, self.start_time, self.max_green,
                                        self.min_green, self.tls_id, self.green_states, self.yellow_states)

    def get_current_state(self):
        return self.traffic_env.get_current_tls_state()

    def reset(self):
        if self.started:
            self.sumo.close()
            self.started = False
        self.start_sim()
        self.ep += 1

        observation = np.zeros((26,), dtype=np.float32)
        return observation

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()

    def __step_ahead(self):
        self.sumo.simulationStep()

    def __do_action(self, action):
        if self.traffic_env.should_do_action:
            self.traffic_env.set_next_phase(action)

    def get_states_code_array(self):
        return self.traffic_env.get_states_code_array()

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self.__step_ahead()
            self.traffic_env.update()
            if self.traffic_env.should_do_action:
                time_to_act = True

    def step(self, action):
        if action is None:
            for _ in range(self.action_frequency):
                self.__step_ahead()
        else:
            self.__do_action(action)
            self._run_steps()
        self.done = self.sim_step >= self.sim_duration
        self.reward = self.__get_reward()

        self.observation = self.__compute_observation()
        return self.observation, self.reward, self.done, self.info

    def __get_reward(self):
        return self.traffic_env.speed_reward()
        # return self.traffic_env.pressure_reward()
        # return self.traffic_env.threshold_waiting_time_reward()
        # return self.traffic_env.threshold_waiting_time_reward_harder_punishment()
        # return self.traffic_env.better_threshold_waiting_time_reward()

    def __compute_observation(self):
        return np.array(self.traffic_env.compute_observation())

    def get_waiting_times_data(self):
        return self.traffic_env.get_waiting_time_data()
