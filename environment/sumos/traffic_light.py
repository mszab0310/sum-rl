traffic_light_states = ['GGGGGrrrrrrrrrrr', 'rrrrGGGGGrrrrrrr', 'rrrrrrrrGGGGGrrr', 'GrrrrrrrrrrrGGGG']
traffic_light_yellow_states = ['yyyyyrrrrrrrrrrr', 'rrrryyyyyrrrrrrr', 'rrrrrrrryyyyyrrr', 'yrrrrrrrrrrryyyy']


class TrafficLight:

    def __init__(self, sumo, gymEnv, action_frequency, start_time, max_green, min_green, tls_id, green_states,
                 yellow_states):
        """ Class representing a traffic light, containing all the associated actions and observations for it
            Args:
                sumo (Sumo): The instance of the Sumo simulation using traci API
                gymEnv (TrafficEnv): The gymnasium environment that the simulation belongs to
                action_frequency (int): The frequency of actions in seconds
                start_time (int): Timestamp in seconds for when to start the traffic light to operate
                max_green (int): Maximum amount of green time per phase
                min_green (int): Minimum amount of green time per phase
                tls_id (str): The id of the traffic light to be controlled from the Sumo instance
                green_states (str[]): A collection of all possible green phases to be used in the controlled junction
                yellow_states (str[]): A collection of all possible yellow phases to be used in the controlled junction
        """
        self.sumo = sumo
        self.gymEnv = gymEnv
        self.action_frequency = action_frequency
        self.start_time = start_time
        self.max_green = max_green
        self.min_green = min_green
        self.tls_id = tls_id
        self.green_states = green_states
        self.yellow_states = yellow_states

        # other fields required for simulation
        self.curr_green = None
        self.yellow_time = 2
        self.is_yellow = False
        self.next_action_time = self.start_time
        self.last_state = None
        self.controlled_links = self.sumo.trafficlight.getControlledLinks(self.tls_id)
        self.min_green_flag = None
        self.state_changed = False
        self.measuring = False
        self.green_time_start = 0
        self.in_lanes = set()
        self.out_lanes = set()
        self.last_phase_change = 0
        self.states_code_array = []
        for edge in self.controlled_links:
            self.in_lanes.add(edge[0][0])
            self.out_lanes.add(edge[0][1])

        self.waiting_time_data = []

    @property
    def should_do_action(self):
        return self.next_action_time == self.gymEnv.sim_step

    def update(self):
        self.last_phase_change += 1
        # sets the actual green phase if it is yellow and enough time has passed
        # print('lp = yelw', self.last_phase_change == self.yellow_time)
        if self.is_yellow and self.last_phase_change == self.yellow_time:
            self.sumo.trafficlight.setRedYellowGreenState(self.tls_id, self.green_states[self.curr_green])
            self.is_yellow = False

    def set_next_phase(self, next_phase_id: int):
        # if new phase same as the last one, or if not enough time since last phase change
        self.last_state = self.curr_green
        if self.curr_green == next_phase_id or self.last_phase_change < self.yellow_time + self.min_green:
            self.states_code_array.append(1)  # used for measuring metrics
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.tls_id, self.green_states[next_phase_id])
            self.next_action_time = self.gymEnv.sim_step + self.action_frequency
        else:
            self.states_code_array.append(-1)  # used for measuring metrics
            self.sumo.trafficlight.setRedYellowGreenState(
                self.tls_id, self.yellow_states[next_phase_id]
            )
            self.curr_green = next_phase_id
            self.next_action_time = self.gymEnv.sim_step + self.action_frequency
            self.is_yellow = True
            self.last_phase_change = 0

    def get_states_code_array(self):
        return self.states_code_array

    # computes the observation space
    def compute_observation(self):
        """Returns an array of 26 values representing the observation space for the reinforcement learning agent
            The values stored in the array come as the following, by their indexes: \n
            0: Decimal value of a one-hot encoding of the current green traffic light state\n
            1-8: The occupancy in % for the last time step on every incoming lane.\n
            9-16: The sum of waiting times (time spent with speed <= 0.1 m/s) of every vehicle, on every incoming lane\n
            17-24: The number of vehicles with a speed <= 0.1 m/s on every incoming lane\n
            25: A 0-1 boolean flag indicating whether min_green time has passed since current green phase was applied or not
        """
        new_state = self.sumo.trafficlight.getRedYellowGreenState(self.tls_id)
        tls_state = get_one_hot_encoding(new_state)
        # tls_state = self.green_states.index(new_state)
        lane_densities = [] * 8
        halting_numbers = [] * 8
        accumulated_waiting_times = [] * 8
        for lane in self.in_lanes:
            lane_densities.append(self.sumo.lane.getLastStepOccupancy(lane))
            halting_numbers.append(self.sumo.lane.getLastStepHaltingNumber(lane))
            vehicles = self.sumo.lane.getLastStepVehicleIDs(lane)
            waiting_time = []
            for vehicle in vehicles:
                waiting_time.append(self.sumo.vehicle.getWaitingTime(vehicle))
            if len(waiting_time) > 0:
                # stores the biggest waiting time only
                accumulated_waiting_times.append(max(waiting_time))
            else:
                accumulated_waiting_times.append(0)
        observation = [tls_state]
        observation.extend(lane_densities)
        observation.extend(accumulated_waiting_times)
        self.waiting_time_data.append(max(accumulated_waiting_times))
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
            self.green_time_start = self.sumo.simulation.getTime()

        if self.measuring and self.sumo.simulation.getTime() - self.green_time_start > self.min_green:
            self.min_green_flag = True

        observation.append(int(self.min_green_flag == True))
        return observation

    def get_current_tls_state(self):
        return get_one_hot_encoding(self.sumo.trafficlight.getRedYellowGreenState(self.tls_id))

    def pressure_reward(self):
        return self.get_lane_pressure()

    def speed_reward(self):
        return self.get_average_speed()

    def get_lane_pressure(self):
        """Computes the difference between the number of vehicles leaving and approaching the intersection"""
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.in_lanes)

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self.get_all_vehicles()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def threshold_waiting_time_reward(self):
        obs = self.compute_observation()

        # Extract relevant values from the observation
        waiting_times = obs[9:17]  # biggest waiting times of each lane in the current time step
        accumulated_waiting_times = sum(waiting_times)
        halting_number = obs[17:25]  # Halting number on each lane
        total_vehicles = sum(halting_number)
        threshold = 90.0
        if total_vehicles == 0 or accumulated_waiting_times == 0:
            reward = 1.0  # Maximum reward when no vehicles are present
        else:
            if accumulated_waiting_times > threshold:
                reward = -1.0 * (accumulated_waiting_times - threshold)  # Negative reward for exceeding the threshold
            else:
                # Positive reward for staying below or equal to the threshold
                reward = (threshold - accumulated_waiting_times)
        return reward

    def better_threshold_waiting_time_reward(self):
        obs = self.compute_observation()
        # Extract relevant values from the observation
        waiting_times = obs[9:17]  # biggest waiting times of each lane in the current time step
        halting_number = obs[17:25]  # Halting number on each lane

        if obs[0] == 1:
            waiting_times[0] = 0
            waiting_times[1] = 0
            halting_number[0] = 0
            halting_number[1] = 0
        if obs[0] == 2:
            waiting_times[2] = 0
            waiting_times[3] = 0
            halting_number[2] = 0
            halting_number[3] = 0
        if obs[0] == 3:
            waiting_times[4] = 0
            waiting_times[5] = 0
            halting_number[4] = 0
            halting_number[5] = 0
        if obs[0] == 4:
            waiting_times[6] = 0
            waiting_times[7] = 0
            halting_number[6] = 0
            halting_number[7] = 0
        accumulated_waiting_times = sum(waiting_times)
        total_vehicles = sum(halting_number)
        threshold = 90.0
        count = len(list(filter(lambda x: x > threshold, waiting_times)))
        reward = 0.5
        if total_vehicles == 0 or accumulated_waiting_times == 0:
            reward += 0.5  # Maximum reward when no vehicles are present
        else:
            if count > len(waiting_times) / 2:
                # if more than 50% of vehicles are waiting for more than threshold punish hard
                reward = -100
                return reward
            if accumulated_waiting_times > threshold:
                reward -= -2.0 * (accumulated_waiting_times - threshold)  # Negative reward for exceeding the threshold
            else:
                # Positive reward for staying below or equal to the threshold
                reward = (threshold - accumulated_waiting_times)
        return reward

    def threshold_waiting_time_reward_harder_punishment(self):
        shouldDone = False
        obs = self.compute_observation()
        # Extract relevant values from the observation
        waiting_times = obs[9:17]  # biggest waiting times of each lane in the current time step
        accumulated_waiting_times = sum(waiting_times)
        print("Waiting times sum: ", accumulated_waiting_times)
        halting_number = obs[17:25]  # Halting number on each lane
        total_vehicles = sum(halting_number)

        threshold = 90.0
        if total_vehicles == 0 or accumulated_waiting_times == 0:
            reward = 1.0  # Maximum reward when no vehicles are present
        else:
            over_time = (lambda arr: any(x > 95 for x in arr))(waiting_times)
            if over_time:
                reward = -500
            elif accumulated_waiting_times > threshold:
                reward = -1.0 * accumulated_waiting_times  # Negative reward for exceeding the threshold
            else:
                # Positive reward for staying below or equal to the threshold
                reward = (threshold - accumulated_waiting_times)
        return reward

    def get_all_vehicles(self):
        veh = []
        for lane in self.in_lanes:
            veh += self.sumo.lane.getLastStepVehicleIDs(lane)
        for lane in self.out_lanes:
            veh += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh

    def get_incoming_vehicles_on_lane(self, lane):
        return self.sumo.lane.getLastStepVehicleIDs(lane)

    def get_all_incoming_vehicles(self):
        veh = []
        for lane in self.in_lanes:
            veh += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh

    def get_waiting_time_data(self):
        return self.waiting_time_data


def get_one_hot_encoding(state):
    """Returns the decimal value for a one-hot encoding of a traffic light state string \n
         0 - current state is not green \n
         [1,4] - corresponding to edges from east to north clockwise
    """
    tls_state = list(state)
    encoding = [0, 0, 0, 0]  # Initialize the encoding with all zeros
    curr_index = 0

    for i in range(0, 13, 4):
        if tls_state[i + 3] == "G":
            encoding[curr_index] = 1
            break
        curr_index += 1
    if 1 in encoding:
        return encoding.index(1) + 1
    else:
        return 0
