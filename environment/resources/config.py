import sys

root = sys.path[1]  # The path of the root directory of the project

# paths to sumo files
base_dir = root + "/environment/simple_junction"
sumo_cfg = base_dir + "/simpleJunction.sumocfg"
trips_file = base_dir + "/trips.trips.xml"
additional_files = base_dir + "/simpleJunction.add.xml"
