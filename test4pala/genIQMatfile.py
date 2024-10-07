import numpy as np
from scipy.io import savemat
from datetime import datetime
import os

# Generate sample IQ data
t = np.arange(0, 1, 0.01)  # Time vector
f = 5  # Signal frequency
I = np.cos(2 * np.pi * f * t)  # In-phase component
Q = np.sin(2 * np.pi * f * t)  # Quadrature component
IQ = I + 1j * Q  # Complex IQ data

# Create a dictionary to store the IQ data
iq_data = {
    't': t,
    'I': I,
    'Q': Q,
    'IQ': IQ
}

# Generate a filename based on the current time
filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.mat'

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate a filename with the prefix 'IQMAT', current time, and '.mat' suffix
filename = 'IQMAT' + datetime.now().strftime("%Y%m%d%H%M%S") + '.mat'

# Combine the directory with the filename
filename = os.path.join(script_dir, filename)

# Save the IQ data to a .mat file
savemat(filename, iq_data)

print(f"IQ data saved to {filename}")