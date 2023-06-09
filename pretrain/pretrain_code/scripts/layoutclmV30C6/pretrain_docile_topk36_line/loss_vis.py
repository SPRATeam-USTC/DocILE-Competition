# Import the libraries
import matplotlib.pyplot as plt
import numpy as np
import json
import os

log_file = "./pretrain.log"

# Read the input file and store the loss and epoch values in lists
input_file = log_file # Change this to your file name
loss_list = []
epoch_list = []
with open(input_file, "r") as f:
    for line in f:
        try:
            data = json.loads(line.strip().replace("'", "\"")) # Try to parse the line as a json string
            loss = data["loss"] # Get the loss value from the json object
            epoch = data["epoch"] # Get the epoch value from the json object
            loss_list.append(loss)
            epoch_list.append(epoch)
        except:
            pass # Skip the line if it is not a valid json string

# Create numpy arrays of the loss and epoch values
loss_array = np.array(loss_list)
epoch_array = np.array(epoch_list)

# Plot the loss line using matplotlib
plt.plot(epoch_array, loss_array)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss line")
plt.savefig(os.path.splitext(log_file)[0]+".png")
