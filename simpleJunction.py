import subprocess
import time

import traci


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


base_dir = "D:/SUMOS/SimpleJunction/"
sumo_cfg = base_dir + "simpleJunction.sumocfg"
trips_file = base_dir + "trips.trips.xml"
additional_files = base_dir + "simpleJunction.add.xml"
sumo_cmd = "sumo"

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
traci.start([sumo_cmd, "-c", sumo_cfg, "-r", trips_file])

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
    print(tls_state, " ", encoding)
    binary_string = ''.join(str(bit) for bit in encoding)
    decimal_number = int(binary_string, 2)
    return decimal_number


edges = {}
controlled_lanes = set()
lanes = traci.trafficlight.getControlledLinks(traffic_light_id)
for lane in lanes:
    edge = lane[0][0]
    controlled_lanes.add(edge)
    key = edge[:-2]
    edges[key] = 0

program_logic = traci.trafficlight.getAllProgramLogics(traffic_light_id)
print(program_logic)

# traci.trafficlight.setRedYellowGreenState(traffic_light_id, "GGGGGGGGGGGGGGGG")
# detector id is e3_number, number in range of [0,7]
# detectors = ["e1_0", "e1_1", "e1_2", "e1_3", "e1_4", "e1_5", "e1_6", "e1_7"]

min_green_time = 15
min_green_flag = False
green_time_start = -1
measuring = False
curr_state = None


def compute_observation_array():
    new_state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
    tls_state = get_one_hot_encoding(new_state)
    lane_densities = []
    lane_queues = []
    for lane in controlled_lanes:
        lane_densities.append(traci.lane.getLastStepOccupancy(lane))
        lane_queues.append(traci.lane.getLastStepHaltingNumber(lane))
    obs_array = [tls_state]
    obs_array.extend(lane_densities)
    obs_array.extend(lane_queues)
    """One hot encoding for tls state and method for measuring green times in simulation seconds"""
    # print(compute_observation_array())

    # if the state has changed
    # if new_state != curr_state:
    #     if min_green_flag:
    #         min_green_flag = False
    #         measuring = False
    #         print("Resetting green flag")
    #         if "G" in new_state:
    #             min_green_flag = False
    #             measuring = False
    #     curr_state = new_state
    #
    # if "G" in new_state and not measuring:
    #     measuring = True
    #     green_time_start = traci.simulation.getTime()
    #
    # if measuring and traci.simulation.getTime() - green_time_start > min_green_time:
    #     print("Min green time exceeded")
    #     min_green_flag = True
    return obs_array


# main loop of simulation
while step < simulation_duration:
    traci.simulationStep()
    step += time_step
    state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
    print(get_one_hot_encoding(state))
    # print(compute_observation_array())
    # for edge in edges:

    #     print("Nr of vehicles on edge ", edge, get_number_of_vehicles_per_edge(edge))
    #     print("Avg speed on edge ", edge, get_average_speed_on_edge(edge))
    #     print("Avg waiting time on edge ", edge, get_average_waiting_times_on_edge(edge))

# fig, axis = plt.subplots(3)
#
# smoothed_values = smooth(density_averages, 0.9)
# axis[0].set_title("Average traffic density")
# axis[0].plot(density_averages)
#
# smoothed_queue_values = smooth(queue_length_averages, 0.9)
# axis[1].set_title("Queue length averages")
# axis[1].plot(smoothed_queue_values)
#
# bin_size = 5
#
# bins = int(np.ceil((max(intersection_waiting_times) - min(intersection_waiting_times)) / bin_size))
#
# axis[2].set_title("Waiting times")
# axis[2].hist(intersection_waiting_times, bins=bins)
#
# fig.tight_layout(pad=0.5)
# plt.show()
# print("Total number of cars ", detector_counter)
end_time = time.time()
print("Run for; ", end_time - start_time)
