import json

json_directory_path = '../mesh_files/'

data = {}

# circuit 1
data['A'] = [[-1, 1]]
data['M'] = len(data['A'])
data['N'] = len(data['A'][0])
data['Y'] = [0.1,0.1]
data['E'] = [10,0]
data['J'] = [0,0]
data['VF'] = [10]
data['IF'] = [0]

with open(json_directory_path + 'test_1' + '.json', 'w') as f:
    json.dump(data, f)

# circuit 2
data['A'] = [[1, 1]]
data['M'] = len(data['A'])
data['N'] = len(data['A'][0])
data['Y'] = [0.1, 0.1]
data['E'] = [0,0]
data['J'] = [10,0]
data['VF'] = [0]
data['IF'] = [10]

with open(json_directory_path + 'test_2' + '.json', 'w') as f:
    json.dump(data, f)

# circuit 3
data['A'] = [[-1,1]]
data['M'] = len(data['A'])
data['N'] = len(data['A'][0])
data['Y'] = [0.1, 0.1]
data['E'] = [10,0]
data['J'] = [0,10]
data['VF'] = [10]
data['IF'] = [5]

with open(json_directory_path + 'test_3' + '.json', 'w') as f:
    json.dump(data, f)

# circuit 4
data['A'] = [[-1, 1, 1, 0], [0, 0, -1, 1]]
data['M'] = len(data['A'])
data['N'] = len(data['A'][0])
data['Y'] = [0.1, 0.1, 0.2, 0.2]
data['E'] = [10,0,0,0]
data['J'] = [0,0,0,10]
data['VF'] = [3]
data['IF'] = [7]

with open(json_directory_path + 'test_4' + '.json', 'w') as f:
    json.dump(data, f)

# circuit 5
data['A'] = [[-1, 1, 0, 0, 1, 0], [0, -1, 1, 1, 0, 0], [0, 0, 0, -1, -1, 1]]
data['M'] = len(data['A'])
data['N'] = len(data['A'][0])
data['Y'] = [0.05, 0.1, 0.033, 0.033, 0.1, 0.033]
data['E'] = [10,0,0,0,0,0]
data['J'] = [0,0,0,0,0,0]
data['VF'] = [5]
data['IF'] = [0]

with open(json_directory_path + 'test_5' + '.json', 'w') as f:
    json.dump(data, f)
