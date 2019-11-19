import os
import argparse
import numpy as np

# Parse base options
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True, default='./',
                    help='Specify target directory of data generation')
parser.add_argument('-r', '--resistor', type=int, required=True, default=None,
                    help='Mesh Resistor Value')
parser.add_argument('-s', '--size', type=int, nargs='+', required=True, default=None,
                    help='Generate permutations for given a list of input channels')
parser.add_argument('-v', '--voltage', type=int, required=False, default=1,
                    help='Mesh Voltage')

# get args
args = vars(parser.parse_args())
directory = args.get('dir')
resistor = args.get('resistor')
size = args.get('permutations')
voltage = args.get('voltage')

# verify args
if resistor <= 0:
	except("Please enter a valid resistance value")
for s in size:
	if s < 2:
		except("Please enter a valid size")

# generate mesh
A = g.generateMesh(size[0], size[1])
A = n.applyVoltage(A)
cols = len(A[0])
    
# generate associated values
R = [resistor for col in range(cols)]
J = [0 for col in range(cols)]
E = [0 for col in range(cols)]
E[-1] = voltage