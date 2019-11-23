import os
import argparse
import numpy as np
from utils import *
import json

# Parse base options
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=False, default='./',
                    help='Specify target directory of data generation')
parser.add_argument('-f', '--filename', type=str, required=False, default='circuit.json',
                    help='Specify target directory of data generation')
parser.add_argument('-r', '--resistor', type=float, required=True, default=None,
                    help='Mesh Resistor Value')
parser.add_argument('-s', '--size', type=int, nargs='+', required=True, default=None,
                    help='Generate permutations for given a list of input channels')
parser.add_argument('-v', '--voltages', type=float, nargs='+', required=True, default=None,
                    help='2 Mesh Voltage')
parser.add_argument('-fq', '--frequencies', type=float, nargs='+', required=True, default=None,
                    help='2 Mesh Frequncies')

# get args
args = vars(parser.parse_args())
directory = args.get('directory')
filename = args.get('filename')
resistor = args.get('resistor')
size = args.get('size')
voltages = args.get('voltages')
frequencies = args.get('frequencies')

print("Directory:", directory)
print("Filename:", filename)
print("Resistor:", resistor)
print("Size:", size)
print("Voltages:", voltages)
print("Frequencies:", frequencies)

if directory[-1] != '/':
    directory = directory + '/'
    
# generate mesh
A = generateMesh(size[0], size[1])
A = applyVoltages(A)
cols = len(A[0])
    
# generate associated values
Y = [1.0 / resistor for col in range(cols)]
J = [0.0 for col in range(cols)]
E = [0.0 for col in range(cols)]
VF = [0.0 for col in range(cols)]
IF = [0.0 for col in range(cols)]
E[-1] = voltages[1]
E[-2] = voltages[0]
VF[-1] = frequencies[1]
VF[-2] = frequencies[0]

# save here A,Y,J,E,VF,IF
data = {}
data['N'] = len(A)
data['M'] = len(A[0])
data['A'] = A
data['Y'] = Y
data['E'] = E
data['J'] = J
data['VF'] = VF
data['IF'] = IF

with open(directory + filename, 'w') as f:
    json.dump(data, f)