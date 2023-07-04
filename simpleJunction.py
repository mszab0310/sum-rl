import subprocess
import time

import matplotlib.pyplot as plt
import openpyxl
import traci

from environment.resources import config


def generate_new_random_traffic(count):
    net_file = "D:/SUMOS/SimpleJunction/simple.net.xml"
    trips_file_output = "D:/SUMOS/SimpleJunction/trips.trips.xml"
    route_file = "D:/SUMOS/SimpleJunction/routes.rou.xml"
    trips_generator = '"c:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py"'
    command = "python " + trips_generator + " -n " + net_file + " -o " + trips_file_output + " -r " + route_file + " -e " + str(
        count) + " --validate"
    subprocess.run(command, shell=True)


def smooth(scalars: [], weight: float) -> []:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


sumo_cmd = "sumo-gui"

# define simulation time steps
step = 0
time_step = 0.0625
simulation_duration = 225

# variables required to store the metrics
density_list = []
all_density = []
density_averages = []

queue_length_list = []
all_queue = []
queue_length_averages = []
intersection_waiting_times = []

start_time = time.time()
# generate_new_random_traffic(5000)
traci.start([sumo_cmd, "-c", config.sumo_cfg, "-r", config.trips_file])

traffic_light_id = "J25"
incoming_lanes = set()


# def calculate_traffic_density(lane_id):
#     lane_length = traci.lane.getLength(lane_id)
#     vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
#
#     occupied_distance = 0
#     for vehicle_id in vehicle_ids:
#         vehicle_pos = traci.vehicle.getLanePosition(vehicle_id)
#         if vehicle_pos <= lane_length:
#             occupied_distance += 1
#     traffic_de = (occupied_distance / lane_length) * 100
#     return traffic_de
#
#


# def measure_queue_length(lane_id):
#     lane_length = traci.lane.getLength(lane_id)
#     vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
#
#     queue_length = sum([traci.vehicle.getLength(vehicle_id) for vehicle_id in vehicles])
#     if queue_length == 0:
#         return 0
#
#     occupancy_percentage = (queue_length * 100) / lane_length
#     return occupancy_percentage
#
#
# # appends waiting times in a given time stamp of all vehicles to an array
# def get_vehicle_waiting_times(vehicle_ids):
#     for veh_id in vehicle_ids:
#         tls_list = traci.vehicle.getNextTLS(veh_id)
#         if len(tls_list) > 0 and tls_list[0][0] == traffic_light_id:
#
#             # Check if the vehicle has come to a full stop at the intersection
#             speed = traci.vehicle.getSpeed(veh_id)
#             if speed < 0.1:
#                 # Get the waiting time of the vehicle at the intersection
#                 waiting_time = traci.vehicle.getWaitingTime(veh_id)
#                 intersection_waiting_times.append(waiting_time)


# number of vehicles per entering edge in a fixed time stamp
def get_number_of_vehicles_per_edge(edge_id):
    vehicle_count = traci.edge.getLastStepVehicleIDs(edge_id)
    return len(vehicle_count)


# returns average speed of vehicles on edge in a fixed time stamp
def get_average_speed_on_edge(edge_id):
    vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
    speeds = []
    for v in vehicles:
        speeds.append(traci.vehicle.getSpeed(v))
    if len(vehicles) > 0:
        return sum(speeds) / len(speeds)
    else:
        return -1


def get_average_waiting_times_on_edge(edge_id):
    vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
    waiting_times = []
    for v in vehicles:
        waiting_times.append(traci.vehicle.getWaitingTime(v))
    if len(vehicles) > 0:
        return sum(waiting_times) / len(waiting_times)
    else:
        return -1


def get_one_hot_encoding(red_yellow_green_state):
    tls_state = list(red_yellow_green_state)
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


in_lanes = set()
out_lanes = set()
lanes = traci.trafficlight.getControlledLinks(traffic_light_id)
for lane in lanes:
    edge = lane[0][0]
    in_lanes.add(edge)
    out_lanes.add(lane[0][1])

program_logic = traci.trafficlight.getAllProgramLogics(traffic_light_id)

# traci.trafficlight.setRedYellowGreenState(traffic_light_id, "GGGGGGGGGGGGGGGG")
# detector id is e3_number, number in range of [0,7]
# detectors = ["e1_0", "e1_1", "e1_2", "e1_3", "e1_4", "e1_5", "e1_6", "e1_7"]

min_green_time = 15
min_green_flag = False
green_time_start = -1
measuring = False
curr_state = None


def compute_observation_array():
    global measuring
    global green_time_start
    new_state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
    tls_state = get_one_hot_encoding(new_state)
    lane_densities = [] * 8
    halting_numbers = [] * 8
    accumulated_waiting_times = [] * 8
    for lane in in_lanes:
        lane_densities.append(traci.lane.getLastStepOccupancy(lane))
        halting_numbers.append(traci.lane.getLastStepHaltingNumber(lane))
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        waiting_time = []
        for vehicle in vehicles:
            waiting_time.append(traci.vehicle.getWaitingTime(vehicle))
        if len(waiting_time) > 0:
            # stores the biggest waiting time only
            accumulated_waiting_times.append(max(waiting_time))
        else:
            accumulated_waiting_times.append(0)
    obs_array = [tls_state]
    obs_array.extend(lane_densities)
    obs_array.extend(accumulated_waiting_times)
    obs_array.extend(halting_numbers)
    """One hot encoding for tls state and method for measuring green times in simulation seconds"""
    # print(compute_observation_array())
    global curr_state
    global min_green_flag
    # if the state has changed
    if new_state != curr_state:
        if min_green_flag:
            min_green_flag = False
            measuring = False
            if "G" in new_state:
                min_green_flag = False
                measuring = False
        curr_state = new_state

    if "G" in new_state and not measuring:
        measuring = True
        green_time_start = traci.simulation.getTime()

    if measuring and traci.simulation.getTime() - green_time_start > min_green_time:
        min_green_flag = True
    obs_array.append(int(min_green_flag == True))
    return obs_array


def get_all_vehicles():
    veh = []
    for lane in in_lanes:
        veh += traci.lane.getLastStepVehicleIDs(lane)
    for lane in out_lanes:
        veh += traci.lane.getLastStepVehicleIDs(lane)
    return veh


def get_lane_pressure():
    """Computes the difference between the number of vehicles leaving and approaching the intersection"""
    return sum(traci.lane.getLastStepVehicleNumber(l) for l in out_lanes) - sum(
        traci.lane.getLastStepVehicleNumber(l) for l in in_lanes)


def get_average_speed() -> float:
    """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

    Obs: If there are no vehicles in the intersection, it returns 1.0.
    """
    avg_speed = 0.0
    vehs = get_all_vehicles()
    if len(vehs) == 0:
        return 1.0
    for v in vehs:
        avg_speed += traci.vehicle.getSpeed(v) / traci.vehicle.getAllowedSpeed(v)
    return avg_speed / len(vehs)


def threshold_waiting_time_reward():
    obs = compute_observation_array()

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


max_times = []
workbook = openpyxl.Workbook()
column_names = ['curr_green', 'occ1', 'occ2', 'occ3', 'occ4', 'occ5', 'occ6', 'occ7', 'occ8',
                'sum1', 'sum2', 'sum3', 'sum4', 'sum5', 'sum6', 'sum7', 'sum8',
                'veh_nr1', 'veh_nr2', 'veh_nr3', 'veh_nr4', 'veh_nr5', 'veh_nr6', 'veh_nr7', 'veh_nr8',
                'min_green_time', 'rew_pressure', 'rew_speed', 'rew_waiting']
sheet = workbook.active
sheet.append(column_names)

# main loop of simulation
while step < simulation_duration:
    traci.simulationStep()
    step += time_step
    data = compute_observation_array()
    max_times.append(max(data[9:17]))
    data.append(get_lane_pressure())
    data.append(get_average_speed())
    data.append(threshold_waiting_time_reward())

    sheet.append(data)

    # state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
    # print(get_one_hot_encoding(state))

    #
    # lanes = list(dict.fromkeys(traci.trafficlight.getControlledLinks(traffic_light_id)))
    # print("-==========lanes===========")
    # print(lanes)
    # edges = {}
    # controlled_lanes = set()
    # lanes = traci.trafficlight.getControlledLinks(traffic_light_id)
    # for lane in lanes:
    #     edge = lane[0][0]
    #     print("IN: ", edge)
    #     print("OUT: ", lane[0][1])
    #     controlled_lanes.add(edge)
    #     key = edge[:-2]
    #     edges[key] = 0
    # print(compute_observation_array())
    # for edge in edges:

    #     print("Nr of vehicles on edge ", edge, get_number_of_vehicles_per_edge(edge))
    #     print("Avg speed on edge ", edge, get_average_speed_on_edge(edge))
    #     print("Avg waiting time on edge ", edge, get_average_waiting_times_on_edge(edge))

workbook.save('observation_data.xlsx')
# fig, axis = plt.subplots(3)
#
# smoothed_values = smooth(density_averages, 0.9)
# axis[0].set_title("Average traffic density")
# axis[0].plot(density_averages)
avg_max = sum(max_times) / len(max_times)
print(avg_max)
plt.plot(max_times)
plt.xlabel("Max waiting times")
plt.ylabel("value count")

plt.show()

print("Max waiting time", max(max_times))
end_time = time.time()
print("Run for; ", end_time - start_time)
