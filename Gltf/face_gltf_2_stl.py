import open3d as o3d
import numpy as np
import json
import base64
from plyfile import PlyData
import io
import matplotlib.pyplot as plt

zoom = 1.5
front = [0.08375353708903234, 0.043816949731044545, -0.99552268680394385]
lookat = [-0.03400410308631812, -0.12749866311234656, 0.48414161276434875]
up = [-0.04105827982788654, 0.99833614468438159, 0.040486539421156605]

def save_and_visualize_selected_points_with_depth_colors(selected_points, output_filename):
    try:
        # Check if the selected points have 'points' attribute
        if not selected_points.has_points():
            print("The selected points do not have any point data.")
            return
        
        # Extract the z-values from the selected points
        z = np.asarray(selected_points.points)[:, 2]

        # Normalize the z-values to get the depth map
        z_normalized = 1 - (z - np.min(z)) / (np.max(z) - np.min(z))

        # Apply a colormap to the normalized z-values to obtain RGB values
        colors = plt.get_cmap('magma')(z_normalized)[:, :3]

        # Assign the colors to the selected points
        selected_points.colors = o3d.utility.Vector3dVector(colors)

        # Save the selected points with the new colors to a PLY file
        o3d.io.write_point_cloud(output_filename, selected_points)

        print(f"Selected points with depth colors saved to '{output_filename}'.")

        # Visualize the saved point cloud
        pcd_to_visualize = o3d.io.read_point_cloud(output_filename)
        o3d.visualization.draw_geometries([pcd_to_visualize],zoom=zoom,front=front, lookat=lookat ,up=up,
                                          window_name="Selected Points Visualization",
                                          point_show_normal=True)

    except Exception as e:
        print(f"An error occurred while saving and visualizing the selected points: {e}")



def visualize_mesh(mesh, zoom, front, lookat, up):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    view_control = vis.get_view_control()
    view_control.set_zoom(zoom)
    view_control.set_front(front)
    view_control.set_lookat(lookat)
    view_control.set_up(up)
    
    # Disable back-face culling
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True

    vis.run()
    vis.destroy_window()

def create_point_cloud_from_ply(ply_data):
    try:
        vertex = ply_data['vertex']
        x, y, z = vertex['x'], vertex['y'], vertex['z']

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        points = np.vstack((x, y, z)).T
        pcd.points = o3d.utility.Vector3dVector(points)

        # Extract normals if present
        nx, ny, nz = vertex['nx'], vertex['ny'], vertex['nz']
        normals = np.vstack((nx, ny, nz)).T
        pcd.normals = o3d.utility.Vector3dVector(normals)

        z_normalized = 1 - (z - np.min(z)) / (np.max(z) - np.min(z))
        colors = plt.get_cmap('magma')(z_normalized)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    except Exception as e:
        print(f"An error occurred while creating the point cloud: {e}")
        return None

try:
    with open("new_entries/train/1/1.gltf", "r") as f:
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

try:
    plydata = PlyData.read(io.BytesIO(decoded_data))
except Exception as e:
    print(f"Error reading PLY data: {e}")
    exit(1)

print(plydata)

pcd = create_point_cloud_from_ply(plydata)

# Create a visualizer object for editing
vis = o3d.visualization.VisualizerWithVertexSelection()
vis.create_window()
vis.add_geometry(pcd)

# Instructions for user
print("1. Shift + left click to pick points.")
print("2. Enter key to finish the selection.")
print("3. Press 'Q' to close the window after you finish the selection.")

vis.run()
vis.destroy_window()

# Retrieve the indices of the selected points
selected_indices = vis.get_picked_points()
selected_indices = [pp.index for pp in vis.get_picked_points()]
#print(selected_indices)

if len(selected_indices) == 0:
    print("No points were selected.")
else:
    # Select the points based on the indices
    selected_points = pcd.select_by_index(selected_indices)
    
    # Check if the original point cloud has colors and normals, then copy them
    if pcd.has_colors():
        selected_points.colors = pcd.select_by_index(selected_indices).colors
    if pcd.has_normals():
        selected_points.normals = pcd.select_by_index(selected_indices).normals

    x_coordinates = np.asarray(selected_points.points)[:, 0]  # All x coordinates
    y_coordinates = np.asarray(selected_points.points)[:, 1]  # All y coordinates
    z_coordinates = np.asarray(selected_points.points)[:, 2]  # All z coordinates

    for x, y, z in zip(x_coordinates, y_coordinates, z_coordinates):
        print(f"X: {x}, Y: {y}, Z: {z}")



    # Make sure normals are present
    if selected_points.has_normals():
        nx_normals = np.asarray(selected_points.normals)[:, 0]  # All nx normal components
        ny_normals = np.asarray(selected_points.normals)[:, 1]  # All ny normal components
        nz_normals = np.asarray(selected_points.normals)[:, 2]  # All nz normal components
        normals = np.vstack((nx_normals, ny_normals ,nz_normals)).T
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        print("Selected points do not have normals. Normals are required for mesh reconstruction.")
        exit(1)


        

    # Check for colors and extract if present
    if selected_points.has_colors():
        red_colors = np.asarray(selected_points.colors)[:, 0]  # All red color components
        green_colors = np.asarray(selected_points.colors)[:, 1]  # All green color components
        blue_colors = np.asarray(selected_points.colors)[:, 2]  # All blue color components
    else:
        print("Selected points do not have colors. Defaulting to gray color for all points.")
        # Default color in case there are no colors in the point cloud
        red_colors = np.full(x_coordinates.shape, 0.5)
        green_colors = np.full(y_coordinates.shape, 0.5)
        blue_colors = np.full(z_coordinates.shape, 0.5)

    # Visualize only the selected part
    o3d.visualization.draw_geometries([selected_points],
                                      zoom=1.5,
                                      front=[0.08375353708903234, 0.043816949731044545, -0.99552268680394385],
                                      lookat=[-0.03400410308631812, -0.12749866311234656, 0.48414161276434875],
                                      up=[-0.04105827982788654, 0.99833614468438159, 0.040486539421156605])




###############################################################

save_and_visualize_selected_points_with_depth_colors(selected_points, 'new_entries/selected_with_depth_colors.ply')



selected_points.estimate_normals()

# Visualize the point cloud with normals
o3d.visualization.draw_geometries([selected_points], point_show_normal=True)

# Orient normals to be consistent
selected_points.orient_normals_consistent_tangent_plane(100)
o3d.visualization.draw_geometries([selected_points], point_show_normal=True)

# Poisson surface reconstruction
print("Running Poisson surface reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    selected_points, depth=9
)
print("Surface reconstruction completed.")

# Visualize the initial mesh
o3d.visualization.draw_geometries([mesh], zoom=zoom, front=front, lookat=lookat, up=up)

# Color mapping based on density
densities = np.asarray(densities)
scaled_densities = (densities - densities.min()) / (densities.max() - densities.min())
density_colors = plt.get_cmap('magma')(scaled_densities)[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)



# Visualize the mesh before removing low-density vertices
visualize_mesh(density_mesh, zoom, front, lookat, up)

# Filter out vertices below a density threshold
threshold = np.quantile(densities, 0.01)
vertices_to_remove = densities < threshold
density_mesh.remove_vertices_by_mask(vertices_to_remove)

# Visualize the mesh after removing low-density vertices
visualize_mesh(density_mesh, zoom, front, lookat, up)


# Compute vertex normals to improve the appearance
density_mesh.compute_vertex_normals()
# Visualize the mesh after removing low-density vertices
visualize_mesh(density_mesh, zoom, front, lookat, up)

# Save the mesh to an STL file
stl_output_file = "output.stl"
print(f"Saving mesh to '{stl_output_file}'...")
try:
    o3d.io.write_triangle_mesh(stl_output_file, density_mesh, write_ascii=False)
    print(f"Mesh saved to '{stl_output_file}' successfully.")
except Exception as e:
    print(f"An error occurred while saving the mesh: {e}")

