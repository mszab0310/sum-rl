import traci
from traci import TraCIException

try:
    connection = traci.getConnection()
    connection.close()
except TraCIException:
    print("No connection ")
