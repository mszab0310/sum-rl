from abc import ABC

import gym
import numpy as np
import traci
from gym import spaces

traffic_light_states = ['GGGGGrrrrrrrrrrr', 'rrrrGGGGGrrrrrrr', 'rrrrrrrrGGGGGrrr', 'GrrrrrrrrrrrGGGG']
traffic_light_yellow_states = ['yyyyyrrrrrrrrrrr', 'rrrryyyyyrrrrrrr', 'rrrrrrrryyyyyrrr', 'yrrrrrrrrrrryyyy']


class TrafficEnv(gym.Env, ABC):

    def __init__(self, sumo_cfg, traffic_light_id, sim_duration):
        super(TrafficEnv, self).__init__()
        self.curr_step = None
        self.obs = None
        self.action_space = spaces.Discrete(4)  # 4 green phase for all edges
        # low and high values for each value from the observation space 18 values in total, element 0 indicates the
        # decimal value of one hot encoding of current traffic light state next 8 values are the % of occupancy of
        # each lane (how much % is occupied by cars) next 8 values are the number of cars on each lane with speed <=
        # 0.1/ms - meaning a full stop or waiting last value is a boolean flag encoded in binary that indicates that
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
        self.sumo_cfg = sumo_cfg
        self.traffic_light_id = traffic_light_id
        self.sim_duration = sim_duration
        self.prev_actions = []
        self.green_time_start = None
        self.min_green_time = 15
        self.measuring = False
        self.curr_state = None
        self.min_green_flag = False
        self.controlled_lanes = None
        self.reward = 0
        self.done = None
        self.start = None
        self.info = None
        self.state_changed = False

    # takes a step in the simulation
    def step(self, action):
        self.prev_actions.append(action)
        traci.simulationStep()
        if self.curr_step > self.sim_duration:
            self.done = True
        self.curr_step += 1
        index = int(action)
        state = traffic_light_states[index]
        traci.trafficlight.setRedYellowGreenState(self.traffic_light_id, state)
        self.obs = self.__get_obs()
        observation = np.array(self.obs)
        self.reward = self.__get_reward()
        self.info = {}

        return observation, self.reward, self.done, self.info

    # computes the observation space
    def __get_obs(self):
        new_state = traci.trafficlight.getRedYellowGreenState(self.traffic_light_id)
        tls_state = get_one_hot_encoding(new_state)
        lane_densities = [] * 8
        lane_queues = [] * 8
        lane_waitings = [] * 8
        for lane in self.controlled_lanes:
            lane_densities.append(traci.lane.getLastStepOccupancy(lane))
            lane_queues.append(traci.lane.getLastStepHaltingNumber(lane))
            lane_waitings.append(traci.lane.getLastStepMeanSpeed(lane))
        observation = [tls_state]
        observation.extend(lane_densities)
        observation.extend(lane_waitings)
        observation.extend(lane_queues)

        # if the state has changed
        if new_state != self.curr_state:
            self.state_changed = True
            if self.min_green_flag:
                self.min_green_flag = False
                self.measuring = False
                if "G" in new_state:
                    self.min_green_flag = False
                    self.measuring = False
            self.curr_state = new_state
        else:
            self.state_changed = False

        if "G" in new_state and not self.measuring:
            self.measuring = True
            self.green_time_start = traci.simulation.getTime()

        if self.measuring and traci.simulation.getTime() - self.green_time_start > self.min_green_time:
            self.min_green_flag = True

        observation.append(int(self.min_green_flag == True))
        return observation

    def reset(self):
        if self.start:
            print("RESET")
            traci.close()
            self.start = False
        self.curr_step = 0
        traci.start(self.sumo_cfg)
        self.state_changed = False
        self.controlled_lanes = set()
        edges = traci.trafficlight.getControlledLinks(self.traffic_light_id)
        for lane in edges:
            edge = lane[0][0]
            self.controlled_lanes.add(edge)
        self.start = True
        self.done = False
        observation = np.zeros((26,), dtype=np.float32)
        return observation

    def __get_reward(self):
        base_reward = -1  # Base reward for each time step

        reward = base_reward
        observation = self.__get_obs()

        # Extract relevant values from the observation
        tls_state = int(observation[0])  # Decimal representation of traffic light state
        occupancy = observation[1:9]  # Traffic occupancy on each lane
        speeds = observation[9:17]  # Traffic occupancy on each lane
        halting_number = observation[17:25]  # Halting number on each lane
        min_green_flag = bool(observation[25])  # Flag indicating if minimum green time has passed

        # Calculate penalties based on occupancy and halting number changes
        for occ, halt, speed in zip(occupancy, halting_number, speeds):
            if occ > 5:
                occupancy_penalty = -0.5 * occ
                reward += occupancy_penalty
            else:
                occupancy_reward = 5
                reward += occupancy_reward

            if halt > 3:
                halting_penalty = -0.5 * halt
                reward += halting_penalty
            else:
                halting_reward = 5
                reward += halting_reward

            if speed < 8.33:
                reward -= 2.5
            else:
                reward += 1.5

        # Penalty for not reaching the minimum green time
        if self.state_changed and not min_green_flag:
            min_green_penalty = -10
            reward += min_green_penalty
        elif not self.state_changed and min_green_flag:
            min_green_reward = 10
            reward += min_green_reward
        elif not min_green_flag:
            reward += 5

        return reward


def get_one_hot_encoding(state):
    tls_state = list(state)
    encoding = [0, 0, 0, 0]  # Initialize the encoding with all zeros
    curr_index = 0

    for i in range(0, 13, 4):
        if tls_state[i + 3] == "G":
            encoding[curr_index] = 1
            break
        curr_index += 1
    binary_string = ''.join(str(bit) for bit in encoding)
    decimal_number = int(binary_string, 2)
    return decimal_number
