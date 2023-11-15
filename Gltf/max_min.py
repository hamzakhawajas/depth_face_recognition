import os

# Define the source directory for depth text files
depth_text_source_dir = "interpolation_depth_numpy"  # Assuming this contains depth text files

# Initialize dictionaries to store min and max x and y values for each person
# min_x_values = {}
# max_x_values = {}
# min_y_values = {}
# max_y_values = {}

# # Iterate through each person
# for person in range(0, 25):  # Assuming persons are numbered from 0 to 24
#     min_x_values[person] = float('inf')
#     max_x_values[person] = -float('inf')
#     min_y_values[person] = float('inf')
#     max_y_values[person] = -float('inf')

#     # Iterate through each file for the current person to find min and max x and y
#     for file_number in range(0, 12):  # Assuming file numbers are from 0 to 11
#         depth_text_filename = f"{str(person).zfill(3)}_{str(file_number).zfill(2)}_face.txt"
#         depth_text_path = os.path.join(depth_text_source_dir, f"{str(person).zfill(3)}", depth_text_filename)

#         if os.path.exists(depth_text_path):
#             with open(depth_text_path, 'r') as f:
#                 lines = f.readlines()

#             for line in lines[1:]:
#                 values = line.strip().split('\t')
#                 x = float(values[0])  # Convert to float
#                 y = float(values[1])  # Convert to float

#                 # Update min and max values
#                 min_x_values[person] = min(min_x_values[person], x)
#                 max_x_values[person] = max(max_x_values[person], x)
#                 min_y_values[person] = min(min_y_values[person], y)
#                 max_y_values[person] = max(max_y_values[person], y)

# # Print the min and max values for each person
# for person in range(0, 25):
#     print(f"Person {person}:")
#     print(f"Min X: {min_x_values[person]}, Max X: {max_x_values[person]}")
#     print(f"Min Y: {min_y_values[person]}, Max Y: {max_y_values[person]}")

# dimensions = {}

# # Iterate through each person
# for person in range(0, 25):  # Assuming persons are numbered from 0 to 24
#     max_width = 0
#     max_height = 0

#     # Iterate through each file for the current person to find the maximum dimensions
#     for file_number in range(0, 12):  # Assuming file numbers are from 0 to 11
#         depth_text_filename = f"{str(person).zfill(3)}_{str(file_number).zfill(2)}_face.txt"
#         depth_text_path = os.path.join(depth_text_source_dir, f"{str(person).zfill(3)}", depth_text_filename)

#         if os.path.exists(depth_text_path):
#             with open(depth_text_path, 'r') as f:
#                 lines = f.readlines()

#             for line in lines[1:]:
#                 values = line.strip().split('\t')
#                 width = int(values[0])  # Convert to integer
#                 height = int(values[1])  # Convert to integer

#                 max_width = max(max_width, width)
#                 max_height = max(max_height, height)

#     dimensions[person] = (max_width, max_height)

# print(dimensions[person])

import os
import numpy as np

# Define the source directory for depth .npy files
depth_npy_source_dir = "new_entries/dataset_numpy_test22"

# Initialize a dictionary to store max and min dimensions for each person
dimensions = {}

# Initialize variables to store the overall max and min dimensions
overall_max_width = -1
overall_max_height = -1
overall_min_width = float('inf')
overall_min_height = float('inf')

# Iterate through each person folder
for person_folder in os.listdir(depth_npy_source_dir):
    person_path = os.path.join(depth_npy_source_dir, person_folder)
    if os.path.isdir(person_path):
        max_width = -1
        max_height = -1
        min_width = float('inf')
        min_height = float('inf')

        # Loop through each file in the person's folder
        for filename in os.listdir(person_path):
            if filename.endswith(".npy"):
                depth_npy_path = os.path.join(person_path, filename)

                # Load the NumPy array and get its shape
                depth_data = np.load(depth_npy_path)
                height, width = depth_data.shape

                # Update the max and min dimensions for this person
                max_width = max(max_width, width)
                max_height = max(max_height, height)
                min_width = min(min_width, width)
                min_height = min(min_height, height)

                # Update the overall max and min dimensions
                overall_max_width = max(overall_max_width, max_width)
                overall_max_height = max(overall_max_height, max_height)
                overall_min_width = min(overall_min_width, min_width)
                overall_min_height = min(overall_min_height, min_height)

        # Store the dimensions for this person
        dimensions[person_folder] = {
            'max': (max_width, max_height),
            'min': (min_width, min_height)

        }

# Print the max and min dimensions for each person
for person_folder in dimensions.keys():
    print(f"Person {person_folder}:")
    print(f"Max Dimensions: {dimensions[person_folder]['max']}")
    print(f"Min Dimensions: {dimensions[person_folder]['min']}")

# Print the overall max and min dimensions
print(f"Overall Max Dimensions: ({overall_max_height}, {overall_max_width})")
print(f"Overall Min Dimensions: ({overall_max_height}, {overall_max_width})")


