import open3d as o3d
import numpy as np
from vedo import *

file_path = "data/point_clouds/pointcloud.npz"

data = np.load(file_path)

print(data.files)

normals = data["normals"].astype(np.float64)

#create open3d pointcloud
vertices = data['points'].astype(np.float64)

# Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud()

# Set the points of the point cloud to the loaded vertices
pcd.points = o3d.utility.Vector3dVector(vertices)
# Attach the normals to the point cloud
pcd.normals = o3d.utility.Vector3dVector(normals)
pcd.paint_uniform_color([0.7, 0.1, 0.1])  # Set point cloud color

print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)

mesh_vedo = Mesh([mesh.vertices, mesh.triangles])

# Show the mesh using Vedo
show(mesh_vedo)
"""
o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])

normals_data /= np.linalg.norm(normals_data, axis=1)[:, np.newaxis]
# Convert the NumPy array into a vedo.Points() object
points = Points(point_cloud)

arrows = Arrows(point_cloud, normals_data, s=0.00001)  # Scale the arrows for visualization




# Create a vedo Plotter
plotter = Plotter()

# Add the point cloud and normals to the plotter
plotter.add(points, arrows)

# Show the plotter
plotter.show()
"""
