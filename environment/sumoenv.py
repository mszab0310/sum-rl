import gym
import numpy
import numpy as np
import traci


class TrafficLightEnv(gym.Env):
    def __init__(self, sumo_cfg, traffic_light_id, sim_duration):
        # initialize the gym environment
        super(TrafficLightEnv, self).__init__()

        # Define the action and observation space
        self.done = None
        self.action_space = gym.spaces.Discrete(4)  # one hot encoding for the 4 edge, each with its on green phase

        # [one hot encoding of current active/last green phase, minimum green flag, lane density 8 values, lane queue
        # 8 values] one hot encoding of 4 values = 0 - nothing, 1 - 2 - 4 - 8 minimum green flag indicates if minimum
        # amount of time has passed for green time (15 seconds usually)
        self.observation_space = gym.spaces.Box(low=-500, high=500, shape=(18,),
                                                dtype=np.float32)  # 18 observation metrics

        # maybe change this to be the whole command for starting the sumo sim take in consideration having 10-50
        # already generated traffic files, the dispersion would be big enough to not load the system with generating
        # them every time
        self.sumo_cfg = sumo_cfg
        self.controlled_lanes = set()
        self.traffic_light_id = traffic_light_id
        self.curr_state = None
        self.min_green_flag = False
        self.measuring = False
        self.green_time_start = -1
        self.min_green_time = 15
        self.step_count = 0
        self.sim_duration = sim_duration

        # Initialize other variables or connections to your simulation environment

    def start_sumo(self):
        traci.start(self.sumo_cfg)
        lanes = traci.trafficlight.getControlledLinks(self.traffic_light_id)
        for lane in lanes:
            edge = lane[0][0]
            self.controlled_lanes.add(edge)

    def _reset(self):
        self.done = False
        self.curr_state = None
        self.min_green_flag = False
        self.measuring = False
        self.step_count = 0
        return self._get_observation()
        # maybe would be good to pick a new random trips file here

    # Reset the simulation environment and return the initial observation

    def step(self, action):
        if self.step_count < self.sim_duration:
            traci.simulationStep()
            self.step_count += 1
        else:
            self.reset()

    # Take an action in the simulation environment
    # Update the simulation based on the action
    # Get the new observation, reward, and done flag

    def _get_observation(self):
        new_state = traci.trafficlight.getRedYellowGreenState(self.traffic_light_id)
        tls_state = get_one_hot_encoding(new_state)
        lane_densities = []
        lane_queues = []
        for lane in self.controlled_lanes:
            lane_densities.append(traci.lane.getLastStepOccupancy(lane))
            lane_queues.append(traci.lane.getLastStepHaltingNumber(lane))
        obs_array = [tls_state]
        obs_array.extend(lane_densities)
        obs_array.extend(lane_queues)

        # if the state has changed
        if new_state != self.curr_state:
            if self.min_green_flag:
                self.min_green_flag = False
                self.measuring = False
                print("Resetting green flag")
                if "G" in new_state:
                    self.min_green_flag = False
                    self.measuring = False
            self.curr_state = new_state

        if "G" in new_state and not self.measuring:
            self.measuring = True
            self.green_time_start = traci.simulation.getTime()

        if self.measuring and traci.simulation.getTime() - self.green_time_start > self.min_green_time:
            print("Min green time exceeded")
            self.min_green_flag = True

        obs_array.append(self.min_green_flag)
        print(obs_array)
        return numpy.asarray(obs_array)

    # Compute and return the current observation based on the state of the simulation

    def _get_reward(self):
        return 10

    # Compute and return the reward based on the current state and action

    def _is_done(self):
        return False


# Check if the episode is done and return a boolean


# Implement any rendering or visualization of the environment if desired
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
    print(decimal_number)
    return decimal_number
