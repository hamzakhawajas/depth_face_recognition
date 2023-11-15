import open3d as o3d

# Load the STL file
mesh = o3d.io.read_triangle_mesh("new_entries/output.stl")

# Check if the mesh is watertight
print("Is the mesh watertight? ", mesh.is_watertight())

# Compute the normal vectors for the mesh
mesh.compute_vertex_normals()

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the mesh to the visualizer
vis.add_geometry(mesh)

# Render the mesh
# Here, we disable back-face culling.
vis.get_render_option().mesh_show_back_face = True

# Run the visualizer
vis.run()
vis.destroy_window()
