import matplotlib.pyplot as plt
import traci

traci.start(
    ["sumo-gui", "-c", "D:/SUMOS/SEMILUNA_VIVO/vivo-semiluna.sumocfg", "-r", "D:/SUMOS/SEMILUNA_VIVO/trips.trips.xml"])

step = 0
time_step = 0.0625
simulation_duration = 225

vehicles = set()
intersection_id = "TL2"
intersection_waiting_times = []
total_vehicles = 0

while step < simulation_duration:
    traci.simulationStep()
    step += time_step

    for veh_id in traci.vehicle.getIDList():
        total_vehicles += 1
        tls_list = traci.vehicle.getNextTLS(veh_id)
        vehicles.add(veh_id)
        if len(tls_list) > 0 and tls_list[0][0] == intersection_id:
            speed = traci.vehicle.getSpeed(veh_id)
            if speed < 0.1:
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                intersection_waiting_times.append(waiting_time)

if intersection_waiting_times:
    avg_waiting_time = sum(intersection_waiting_times) / len(intersection_waiting_times)
    print(f"Average waiting time at intersection {intersection_id}: {avg_waiting_time:.2f} seconds")
    print(f"Car count with waiting time under 16 s {len([x for x in intersection_waiting_times if x < 2.0])}")
    print(f"Car count with waiting time above 60 s {len([x for x in intersection_waiting_times if x > 60.00])}")
    print(f"with a total vehicle count at intersection of {len(intersection_waiting_times)}")
    print(f"Total vehicles in simulation {total_vehicles}")
    print(f"Total departed {len(vehicles)}")
else:
    print(f"No vehicles passed through intersection {intersection_id}")

bin_size = 5
num_bins = int((max(intersection_waiting_times) - min(intersection_waiting_times)) / bin_size) + 1

plt.hist(intersection_waiting_times, bins=num_bins,
         range=(min(intersection_waiting_times), max(intersection_waiting_times)), edgecolor='black')
plt.xlabel("Waiting time (seconds)")
plt.ylabel("Vehicle count")
plt.text(0.6, 0.95, f"Total vehicles: {len(intersection_waiting_times)}", transform=plt.gca().transAxes)
plt.title("Waiting times")
plt.show()
