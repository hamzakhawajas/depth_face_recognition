import os
import numpy as np

# Define the source and destination directories
source_dir = "final_depth_numpy"
dest_dir = "final_depth_numpy_clean"

# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop over each person's folder
for person_folder in os.listdir(source_dir):
    person_path = os.path.join(source_dir, person_folder)
    
    # Create corresponding folder in the destination directory
    dest_person_path = os.path.join(dest_dir, person_folder)
    if not os.path.exists(dest_person_path):
        os.makedirs(dest_person_path)

    # Check if it's a folder
    if os.path.isdir(person_path):
        # Loop over each .npy file in person's folder
        for item_file in os.listdir(person_path):
            if item_file.endswith('.npy'):
                item_path = os.path.join(person_path, item_file)
                # Load the numpy array
                depth_array = np.load(item_path)

                # Convert NaN values to 0
                depth_array = np.nan_to_num(depth_array, nan=0.0)

                # Save the cleaned array into the destination folder
                dest_item_path = os.path.join(dest_person_path, item_file)
                np.save(dest_item_path, depth_array)

print("Cleaning completed!")
