import os
import numpy as np
def print_padded_arrays(output_folder):
    count=0
    for root, _, files in os.walk(output_folder):
        #print(files)
        for file in files:
            if file.endswith('.npy'):
                padded_depth_data = np.load(os.path.join(root, file))
                print(f'Array from File: {os.path.join(root, file)}\n', padded_depth_data)
                print(f'Shape: {padded_depth_data.shape}\n')
                #print(file)
                #print(count)
                count+=1

    print(count)

if __name__ == "__main__":
    output_folder = "downsampled_depth_numpy"  # Replace with the actual output folder path
    print_padded_arrays(output_folder)
