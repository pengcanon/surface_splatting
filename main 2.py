import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import hdbscan 
import pymeshlab as ml
import time
import subprocess


# Record the start time
start_time = time.time()

'''
ms_preprocess = ml.MeshSet()
ms_preprocess.load_new_mesh("./Meshes/point_cloud_000000.ply")
ms_preprocess.apply_filter('generate_simplified_point_cloud')
ms_preprocess.save_current_mesh("./Output/preprocess.ply")
'''

# Load the point cloud
pcd = o3d.io.read_point_cloud("./Meshes/point_cloud_000000.ply")

# Convert Open3D.o3d.geometry.PointCloud to numpy array
vertices = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# DBSCAN clustering
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=50, min_points=5, print_progress=True))

# Create an empty mesh to store the final result
final_meshset = ml.MeshSet()

# Iterate over each cluster
unique_labels = np.unique(labels)
i=0
for label in unique_labels:
    if label == -1:
        continue  # Skip noise (label -1)

    print(f"Processing label: {label}")

    # Extract points of the current cluster
    cluster_points = vertices[labels == label]
    cluster_colors = colors[labels == label]

    if len(cluster_points)<15000:
        continue

    # Create a point cloud for the current cluster
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

    # Save the point cloud to a file
    o3d.io.write_point_cloud("cluster.ply", cluster_pcd)

    # Load the point cloud into MeshLab
    ms = ml.MeshSet()
    ms.add_mesh(
        ml.Mesh(
            vertex_matrix=cluster_points, v_color_matrix=np.hstack([cluster_colors, np.ones((len(cluster_colors), 1))])
        )
    )
    
   # ms.apply_filter('generate_simplified_point_cloud')
    #Generate Normals and Reconstruct Mesh
    ms.compute_normal_for_point_clouds(k=50) #default is 10
    #ms.generate_marching_cubes_rimls()
    ms.generate_surface_reconstruction_screened_poisson()

    #Generate Texture for each cluster Testing: [Comment out if not needed]
 
    ms.apply_filter("compute_texcoord_by_function_per_vertex")
    ms.apply_filter("compute_texcoord_transfer_vertex_to_wedge")
    ms.apply_filter("compute_texcoord_parametrization_triangle_trivial_per_wedge", textdim=4096, border = 2, method='Space-optimizing')
    ms.apply_filter("compute_texmap_from_color", textname=f"Processing{label}.png", textw=4096, texth=4096)
    ms.save_current_mesh(f"./Output/Single/Processing{label}.obj", save_face_color=True, save_wedge_texcoord=True, save_wedge_normal=False)

 
# Record the end time
end_time = time.time()

# Calculate the elapsed time, in seconds
elapsed_time = end_time - start_time

print(f"The code took {elapsed_time} seconds to run.")

blender_executable_path = "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe"

# Specify the path to the Python script
script_path = "./blender.py"

# Run the command
#subprocess.run([blender_executable_path, "--background", "--python", script_path])

'''
    # Get the resulting mesh
    mesh = ms.current_mesh()


    # Add the resulting mesh to the final MeshSet
    final_meshset.add_mesh(mesh, "mesh" + str(label))

final_meshset.apply_filter('generate_by_merging_visible_meshes')

#Generate Texture
#final_meshset.compute_normal_for_point_clouds()
final_meshset.apply_filter("compute_texcoord_by_function_per_vertex")
final_meshset.apply_filter("compute_texcoord_transfer_vertex_to_wedge")
final_meshset.apply_filter("compute_texcoord_parametrization_triangle_trivial_per_wedge", textdim=4096, border = 0, method='Space-optimizing')
final_meshset.apply_filter("compute_texmap_from_color", textname="texture.png", textw=4096, texth=4096)

# Save the final mesh to a file
final_meshset.save_current_mesh("./Output/poisson_mesh.obj", True)
print("Final mesh saved to poisson_mesh.obj")

# Record the end time
end_time = time.time()

# Calculate the elapsed time, in seconds
elapsed_time = end_time - start_time

print(f"The code took {elapsed_time} seconds to run.")
'''