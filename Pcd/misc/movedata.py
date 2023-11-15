import os
import shutil

# Source directory containing the numpy files
source_dir = "padded_depth_numpy/"

# Destination directory where you want to move all the numpy files
destination_dir = "all_numpy_files/"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Iterate through the source directory and subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".npy"):
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_dir, file)
            shutil.move(source_file, destination_file)
            print(f"Moved {source_file} to {destination_file}")

print("All numpy files have been moved to the destination directory.")
