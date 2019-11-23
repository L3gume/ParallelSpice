import numpy as np
import json
import matplotlib.pyplot as plt

#### change path to desired output
json_path = '../mesh_files/test_output.json'

#### change node to desired node
node = 0

def update_annot(ind):
    x,y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = ("Voltage = " + str('{:.2E}'.format(y[ind["ind"][0]])) + "V, Time = " + str('{:.2E}'.format(x[ind["ind"][0]])) + "s")
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = line.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


with open(json_path, 'r') as f:
    json_dict = json.load(f)

timeSteps = 0
voltages = []

for matrix in json_dict:
    # if json element with b, it is a matrix with the associated voltages
    if matrix.startswith('entries'):
        entries = json_dict[matrix]
        for entry_t in entries:
            voltages_at_timestep = []
            for node_id in range(len(entry_t)):
                # add to the voltages to plot
                voltages_at_timestep.append(entry_t[node_id])
            voltages.append(voltages_at_timestep)
    # if json element is timestep, then we have the timestep
    elif matrix.startswith('timestep'):
        # add to the time step
        timeSteps = json_dict[matrix]

# combine by node rather than by timestep
voltages = np.column_stack(voltages)

x = np.arange(0, timeSteps*len(voltages[node]), timeSteps)
y = voltages[node]
fig, ax = plt.subplots()
line, = plt.plot(x, y)
annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"))
annot.set_visible(True)

plt.title('Voltage at node ' + str(node))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
