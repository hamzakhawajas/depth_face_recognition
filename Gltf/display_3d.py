import open3d as o3d
import tempfile
import os
import json
import base64
from plyfile import PlyData
import io

try:
    with open("new_entries/train/5/5.gltf", "r") as f:
        gltf_data = json.load(f)
except Exception as e:
    print(f"Error loading GLTF: {e}")
    exit(1)

try:
    data_uri = gltf_data["meshes"][0]["extras"]["dataURI"]
except (KeyError, IndexError) as e:
    print(f"Couldn't find the necessary data in the GLTF file: {e}")
    exit(1)

base64_data = data_uri.split(",")[-1]
decoded_data = base64.b64decode(base64_data)
buffer = io.BytesIO(decoded_data)
try:
    plydata = PlyData.read(buffer)
except Exception as e:
    print(f"Error reading PLY data: {e}")
print(plydata)

# Create a temporary file to store the PLY data
with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as temp_file:
    temp_file.write(decoded_data)
    temp_ply_file_name = temp_file.name

pcd = o3d.io.read_point_cloud(temp_ply_file_name)

o3d.visualization.draw_geometries(
    [pcd],
    zoom=0.69999999999999996,
    front=[ 0.08375353708903234, 0.043816949731044545, -0.99552268680394385 ],
	lookat=[ -0.03400410308631812, -0.12749866311234656, 0.48414161276434875 ],
	up=[ -0.04105827982788654, 0.99833614468438159, 0.040486539421156605 ],
)
os.remove(temp_ply_file_name)
