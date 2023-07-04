import random
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
        self.sumo_cfg = sumo_cfg
        self.traffic_light_id = traffic_light_id
        self.sim_duration = sim_duration
        self.prev_actions = []
        self.green_time_start = None
        self.min_green_time = 15
        self.measuring = False
        self.last_state = None
        self.curr_state = random.choice(traffic_light_states)
        self.min_green_flag = False
        self.controlled_lanes = None
        self.reward = 0
        self.done = None
        self.start = None
        self.info = None
        self.state_changed = False
        self.last_waiting_time_sum = 0.0
        self.last_action = 0
        self.action_period = 10
        self.vehicles = dict()

    # takes a step in the simulation
    def step(self, action):
        self.prev_actions.append(action)
        traci.simulationStep()
        if self.curr_step > self.sim_duration:
            self.done = True
        self.curr_step += 1
        # makes the agent make an action after action_period amount of time
        # if not then persist the current state
        if self.curr_step - self.last_action >= self.action_period:
            print("action")
            index = int(action)
            state = traffic_light_states[index]
            traci.trafficlight.setRedYellowGreenState(self.traffic_light_id, state)
            self.last_action = self.curr_step
            self.curr_state = state
        else:
            print("not action")
            traci.trafficlight.setRedYellowGreenState(self.traffic_light_id, self.curr_state)
        self.obs = self.__get_obs()
        observation = np.array(self.obs)
        self.reward = self.__get_reward()
        self.info = {}

        return observation, self.reward, self.done, self.info

    # computes the observation space
    def __get_obs(self):
        new_state = traci.trafficlight.getRedYellowGreenState(self.traffic_light_id)
        # tls_state = get_one_hot_encoding(new_state)
        tls_state = get_state_index(new_state)
        lane_densities = [] * 8
        halting_numbers = [] * 8
        accumulated_waiting_times = [] * 8
        for lane in self.controlled_lanes:
            lane_densities.append(traci.lane.getLastStepOccupancy(lane))
            halting_numbers.append(traci.lane.getLastStepHaltingNumber(lane))
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            waiting_time = []
            for vehicle in vehicles:
                waiting_time.append(traci.vehicle.getWaitingTime(vehicle))
            if len(waiting_time) > 0:
                accumulated_waiting_times.append(sum(waiting_time))
            else:
                accumulated_waiting_times.append(0)
        observation = [tls_state]
        observation.extend(lane_densities)
        observation.extend(accumulated_waiting_times)
        observation.extend(halting_numbers)

        # if the state has changed
        if new_state != self.last_state:
            self.state_changed = True
            if self.min_green_flag:
                self.min_green_flag = False
                self.measuring = False
                if "G" in new_state:
                    self.min_green_flag = False
                    self.measuring = False
            self.last_state = new_state
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
            traci.close()
            self.start = False
        self.curr_step = 0
        traci.start(self.sumo_cfg)
        self.state_changed = False
        self.controlled_lanes = set()
        edges = traci.trafficlight.getControlledLinks(self.traffic_light_id)
        self.vehicles = traci.vehicle.getIDList()
        for lane in edges:
            edge = lane[0][0]
            self.controlled_lanes.add(edge)
        self.start = True
        self.done = False
        observation = np.zeros((26,), dtype=np.float32)
        return observation

    def __get_reward(self):
        base_reward = 0  # Base reward for each time step

        reward = base_reward
        observation = self.__get_obs()

        tls_state = int(observation[0])  # Decimal representation of traffic light state
        occupancy = observation[1:9]  # Traffic occupancy on each lane
        # print(print_lane_pairs(occupancy))
        waiting_times = observation[9:17]  # Traffic occupancy on each lane in percentage
        halting_number = observation[17:25]  # Halting number on each lane
        min_green_flag = bool(observation[25])  # Flag indicating if minimum green time has passed

        # for occ, halt, wait in zip(occupancy, halting_number, waiting_times):
        #     if occ > 0.8:
        #         occupancy_penalty = -0.5 * occ
        #         reward += -60
        #     else:
        #         occupancy_reward = 60
        #         reward += occupancy_reward
        #
        #     if wait > 90:
        #         reward += -25
        #     else:
        #         reward += 25
        #
        #     if halt > 20:
        #         halting_penalty = -0.5 * halt
        #         reward += - 15
        #     else:
        #         halting_reward = 15
        #         reward += halting_reward

        # if self.state_changed and not min_green_flag:
        #     min_green_penalty = -10
        #     reward += min_green_penalty
        # elif not self.state_changed and min_green_flag:
        #     min_green_reward = 10
        #     reward += min_green_reward
        # elif not min_green_flag:
        #     reward += 5

        wait_ts = sum(waiting_times) / 100.0
        print(str(self.last_waiting_time_sum) + " - " + str(wait_ts))
        reward = self.last_waiting_time_sum - wait_ts
        self.last_waiting_time_sum = wait_ts
        # if reward > 0:
        #     print(reward)
        # else: print("-")
        return reward

    def get_accumulated_waiting_time_per_lane(self):
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.controlled_lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.vehicles:
                    self.vehicles[veh] = {veh_lane: acc}
                else:
                    self.vehicles[veh][veh_lane] = acc - sum(
                        [self.vehicles[veh][lane] for lane in self.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane


def get_state_index(state):
    return traffic_light_states.index(state)


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


def print_lane_pairs(edges):
    edge_names = ["East", "South", "West", "North"]
    lane_pairs = [(edges[i], edges[i + 1]) for i in range(0, 8, 2)]

    for i, pair in enumerate(lane_pairs):
        edge_name = edge_names[i % 4]
        # print(f"Lanes for {edge_name} edge: {pair[0]}, {pair[1]}")



