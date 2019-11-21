import os
import numpy as np
import json
import matplotlib.pyplot as plt

json_path = '../mesh_files/test.json'

with open(json_path, 'r') as f:
    json_dict = json.load(f)

# we want to plot the voltages of nodes:
node_ids = [0, 1, 2]

timeSteps = 0
voltages = []

for matrix in json_dict:
    # if json element with b, it is a matrix with the associated voltages
    if matrix.startswith('b'):
        b = json_dict[matrix]
        voltages_at_timestep = []
        for node in node_ids:
            # add to the voltages to plot
            voltages_at_timestep.append(b[node])
        voltages.append(voltages_at_timestep)
    # if json element is timestep, then we have the timestep
    elif matrix.startswith('timestep'):
        # add to the time step
        timeSteps = json_dict[matrix]

# combine by node rather than by timestep
voltages = np.column_stack(voltages)

# plot each node graph individually
for node in node_ids:
    x = np.arange(0, timeSteps*len(voltages[node]), timeSteps)
    y = voltages[node]
    plt.plot(x, y)
    plt.title('Voltage at node ' + str(node))
    plt.xlabel('sample(n)')
    plt.ylabel('voltage(V)')
    plt.show()
