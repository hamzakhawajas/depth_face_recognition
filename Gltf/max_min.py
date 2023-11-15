import os
import numpy as np

# depth_text_source_dir = "interpolation_depth_numpy"
depth_npy_source_dir = "dataset_numpy_train"
dimensions = {}

overall_max_width = -1
overall_max_height = -1
overall_min_width = float('inf')
overall_min_height = float('inf')

for person_folder in os.listdir(depth_npy_source_dir):
    person_path = os.path.join(depth_npy_source_dir, person_folder)
    if os.path.isdir(person_path):
        max_width = -1
        max_height = -1
        min_width = float('inf')
        min_height = float('inf')

        for filename in os.listdir(person_path):
            if filename.endswith(".npy"):
                depth_npy_path = os.path.join(person_path, filename)

                depth_data = np.load(depth_npy_path)
                height, width = depth_data.shape

                max_width = max(max_width, width)
                max_height = max(max_height, height)
                min_width = min(min_width, width)
                min_height = min(min_height, height)

                overall_max_width = max(overall_max_width, max_width)
                overall_max_height = max(overall_max_height, max_height)
                overall_min_width = min(overall_min_width, min_width)
                overall_min_height = min(overall_min_height, min_height)

        dimensions[person_folder] = {
            'max': (max_width, max_height),
            'min': (min_width, min_height)

        }

for person_folder in dimensions.keys():
    print(f"Person {person_folder}:")
    print(f"Max Dimensions: {dimensions[person_folder]['max']}")
    print(f"Min Dimensions: {dimensions[person_folder]['min']}")

print(f"Overall Max Dimensions: ({overall_max_height}, {overall_max_width})")
print(f"Overall Min Dimensions: ({overall_max_height}, {overall_max_width})")


