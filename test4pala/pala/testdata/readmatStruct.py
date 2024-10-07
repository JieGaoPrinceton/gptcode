import os
from scipy.io import loadmat
import numpy as np


def read_and_print_mat_struct(filename):
    # Load the .mat file
    mat_data = loadmat(filename)
    
    # Print all the information contained in the structure
    for key, value in mat_data.items():
        if key.startswith('__'):
            continue
        print("{key}:")
        if isinstance(value, np.ndarray) or np.isscalar(value):
            print(np.array2string(value, separator=', '))
        elif isinstance(value, (dict, list)):
            print(value)
        else:
            print(value)
        print("\n")

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Example usage
filename = os.path.join(current_dir, 'PALA_InVivoMouseTumor_001.mat')  # Replace with your actual .mat file name
read_and_print_mat_struct(filename)