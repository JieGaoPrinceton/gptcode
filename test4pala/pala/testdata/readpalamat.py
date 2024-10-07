import numpy as np
from scipy.io import loadmat
import datetime
import os

def read_and_display_iq_data(filename):
    # Load the .mat file
    mat_data = loadmat(filename)
    
    # Extract the I and Q components
    I = mat_data['I'].flatten()
    Q = mat_data['Q'].flatten()
    
    # Reconstruct the IQ data
    IQ = I + 1j * Q
    
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"matout_{timestamp}.txt"

    # Write the first 20 rows of the IQ data to a text file
    with open(output_filename, 'w') as f:
        for i in range(min(20, len(IQ))):
            f.write(f"Row {i+1}: I = {I[i]}, Q = {Q[i]}, IQ = {IQ[i]}\n")

# Example usage
directory = os.path.dirname(__file__)
filename = os.path.join(directory, 'PALA_InVivoMouseTumor_001.mat')  # Replace with your actual .mat file name
read_and_display_iq_data(filename)