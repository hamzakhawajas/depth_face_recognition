import open3d as o3d
import tempfile
import os
import numpy as np
import json
import base64
from plyfile import PlyData
import io
import matplotlib.pyplot as plt

# Function to create a point cloud from PLY data with inverted depth colors
def create_point_cloud_from_ply(ply_data):
    # Extract vertex data
    vertex = ply_data['vertex']
    # Extract x, y, z coordinates
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    # Combine x, y, z coordinates into a single array and assign to the point cloud
    points = np.vstack((x, y, z)).T
    pcd.points = o3d.utility.Vector3dVector(points)

    # Normalize z values to 0-1 for color mapping and invert the colors
    z_normalized = 1 - (z - np.min(z)) / (np.max(z) - np.min(z))
    # Get colors from inverted magma colormap
    colors = plt.get_cmap('magma')(z_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

# Load GLTF file and extract PLY data
try:
    with open("new_entries/train/5/3.gltf", "r") as f:
        gltf_data = json.load(f)
except Exception as e:
    print(f"Error loading GLTF: {e}")
    exit(1)

try:
    data_uri = gltf_data["meshes"][0]["extras"]["dataURI"]
except (KeyError, IndexError) as e:
    print(f"Couldn't find the necessary data in the GLTF file: {e}")
    exit(1)

# Decode the base64-encoded binary data
base64_data = data_uri.split(",")[-1]
decoded_data = base64.b64decode(base64_data)

# Read the PLY data from the decoded binary data
try:
    plydata = PlyData.read(io.BytesIO(decoded_data))
except Exception as e:
    print(f"Error reading PLY data: {e}")
    exit(1)

# Create a point cloud from the PLY data with inverted depth colors
pcd = create_point_cloud_from_ply(plydata)

# Visualize the point cloud with inverted depth colors
o3d.visualization.draw_geometries(
    [pcd],
    zoom=0.69999999999999996,
    front=[0.08375353708903234, 0.043816949731044545, -0.99552268680394385],
    lookat=[-0.03400410308631812, -0.12749866311234656, 0.48414161276434875],
    up=[-0.04105827982788654, 0.99833614468438159, 0.040486539421156605]
)
