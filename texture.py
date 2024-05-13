import os
import PIL
import time

import cupy as cp
import numpy as np
import open3d as o3d


def generate_texture(
    input_pcl: str | o3d.geometry.PointCloud,
    input_mesh: str | o3d.geometry.TriangleMesh,
    generate_texture_coords: bool = True,
    attach_texture_to_mesh: bool = True,
    save_obj: bool = True,
    include_texcoords_in_obj: bool = True,
    output_obj_path: str = None,
    save_texture: bool = True,
    output_texture_path: str = None,
    texture_size: int = 2048,
    texture_divisions: int = 1,
    lut: PIL.ImageFilter = None,
    nearest_neighbors: int = 3,
    buffer_pct: float = 25.0,
    verbose: bool = True,
) -> tuple[o3d.geometry.TriangleMesh, cp.ndarray, list[int]]:
    """
    Generates a texture for a given point cloud and mesh.

    Args:
        input_pcl (str | o3d.geometry.PointCloud): The path to the input point cloud file or the actual point cloud object.
        input_mesh (str | o3d.geometry.TriangleMesh): The path to the input mesh file or the actual mesh object.
        save_obj (bool, optional): Whether to save the output mesh to an OBJ file. Defaults to True.
        include_texcoords (bool, optional): Whether to include texture coordinates in the saved obj. Defaults to True.
        output_mesh_path (str, optional): The path to save the output mesh file if save_obj = True. Defaults to None.
        save_texture (bool, optional): Whether to save the output texture to an image file. Defaults to True.
        output_texture_path (str, optional): The path to save the output texture file if save_texture = True. Defaults to None.
        texture_size (int, optional): The size of the output texture (width and height). Defaults to 2048.
        texture_divisions (int, optional): The texture will be split into a grid of this many bins on each size, with largest triangles in the top left. Defaults to 1, so no divisions.
        nearest_neighbors (int, optional): The number of nearest neighbors to consider for color averaging. Defaults to 3.
        buffer_pct (float, optional): The percentage of buffer to add to the texture coordinates to avoid edge bleeding. Defaults to 25%.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        tuple[o3d.geometry.TriangleMesh, cp.ndarray, list[int]]: A tuple containing the generated triangle mesh,
        the texture as a CuPy array, and a list of the number of triangles in each bin.
    """
    # Load the original point cloud
    pcd = o3d.io.read_point_cloud(input_pcl) if isinstance(input_pcl, str) else input_pcl
    original_points = cp.asarray(pcd.points, dtype=cp.float32)
    original_colors = cp.asarray(pcd.colors, dtype=cp.float32)
    if verbose:
        print(pcd)

    # Load the mesh from blender
    mesh = o3d.io.read_triangle_mesh(input_mesh) if isinstance(input_mesh, str) else input_mesh
    vertices = cp.asarray(mesh.vertices, dtype=cp.float32)
    if verbose:
        print(mesh)

    start_time = time.perf_counter()

    # The meshes that come out of blender are kind of "puffy", so first move mesh vertices toward their nearest
    # point cloud neighbor; this shrinks the mesh to better fit the point cloud
    start_move_vertices = time.perf_counter()
    _, idx = KNN_lookup_gpu(original_points, vertices, 1)
    vertices = original_points[idx.flatten()]
    # add moved vertices back to mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices.get())
    if verbose:
        print(
            f"Moved mesh vertices to nearest point cloud neighbors, time: {time.perf_counter() - start_move_vertices:.4f} seconds"
        )

    # delete duplicated vertices, update triangles
    start_remove_duplicated_vertices = time.perf_counter()
    old_n_vertices = len(vertices)
    old_n_triangles = len(mesh.triangles)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    vertices = cp.asarray(mesh.vertices, dtype=cp.int16)  # use int16 to force lossless encoding in Draco
    triangles = cp.asarray(mesh.triangles, dtype=cp.int32)
    if verbose:
        print(
            f"Removed {old_n_vertices - len(vertices)} duplicated vertices, {old_n_triangles - len(triangles)} degenerate triangles, "
            + f"time: {time.perf_counter() - start_remove_duplicated_vertices:.4f} seconds"
        )

    # sort vertices within triangles to avoid reordering by Draco
    start_sort_vertices = time.perf_counter()
    triangles = sort_triangle_vertices(triangles, vertices)
    # remove any faces that are permuted versions of other faces
    triangles, removed_indices = remove_permuted_faces(triangles)
    if verbose:
        print(
            f"Removed {len(removed_indices)} permuted faces, new total: {len(triangles)}, "
            + f"time: {time.perf_counter() - start_sort_vertices:.4f} seconds"
        )

    # create a texture_size x texture_size array split into texture_divisions x texture_divisions squares;
    # each square will contain the triangles that make up that portion of the total triangle area, and they will be
    # numbered within each square from largest to smallest, row by row.
    # Each pixel will contain the triangle number for that pixel in the texture (will be -1 if that pixel is not in a triangle)
    start_sort_triangles = time.perf_counter()
    n_triangles = len(triangles)
    max_per_bin = calculate_max_triangles_per_bin(texture_size, texture_divisions)
    split_indices, split_areas = split_triangles_into_equal_area_bins(
        triangles,
        vertices,
        n_bins=texture_divisions**2,
        max_per_bin=max_per_bin,
        return_sizes=True,
        remove_small=True,
        area_threshold_percent=0.01,
    )
    sorted_indices = cp.hstack(split_indices)
    triangles = triangles[sorted_indices]
    n_triangles_list = [len(triangles) for triangles in split_indices]
    if verbose:
        print(
            f"Sorted triangles by area, removed {n_triangles - sum(n_triangles_list)} small triangles, "
            + f"new total: {sum(n_triangles_list)}, time: {time.perf_counter() - start_sort_triangles:.4f} seconds"
        )
    n_triangles = sum(n_triangles_list)

    # preallocate arrays for overall triangle numbers and valid pixel mask
    start_triangle_numbers = time.perf_counter()
    triangle_numbers, valid_px_mask, triangle_sizes_list, grid_sizes_list = generate_triangle_numbers_grid(
        n_triangles_list, texture_size, texture_divisions, verbose=verbose
    )
    if verbose:
        print(
            f"Got triangle numbers, shape: {triangle_numbers.shape}, dtype: {triangle_numbers.dtype}, time: {time.perf_counter() - start_triangle_numbers:.4f} seconds"
        )

    # for each pixel, determine the UV coordinates of the vertices of the triangle it is in
    # output is shape (texture_size, texture_size, 3, 2) and will contain NaNs for pixels not in a triangle
    start_uv_coords = time.perf_counter()
    triangle_uv_coords = generate_triangle_uv_vertex_coords_grid(
        n_triangles_list, triangle_sizes_list, grid_sizes_list, texture_size, texture_divisions
    )
    # add a buffer to the UV coordinates to avoid edge bleeding
    triangle_uv_coords = buffer_uv_coords_barycenter(triangle_uv_coords, buffer_pct=buffer_pct)
    uv_coords = cp.full((texture_size, texture_size, 3, 2), cp.nan, dtype=cp.float32)
    uv_coords[valid_px_mask] = triangle_uv_coords[triangle_numbers[valid_px_mask]]
    if verbose:
        print(
            f"Got UV coords, shape: {uv_coords.shape}, dtype: {uv_coords.dtype}, time: {time.perf_counter() - start_uv_coords:.4f} seconds"
        )

    # determine weights as barycentric coordinates of each pixel in its triangle
    # output is shape (texture_size, texture_size, 3) and will contain NaNs for pixels not in a triangle
    start_xyz = time.perf_counter()
    ij_matrix = cp.indices((texture_size, texture_size)).transpose(1, 2, 0)  # shape (texture_size, texture_size, 2)
    weights = cp.linalg.solve(
        cp.concatenate([uv_coords, cp.ones_like(uv_coords[..., :1])], axis=-1).transpose(0, 1, 3, 2),
        cp.concatenate([ij_matrix, cp.ones_like(ij_matrix[..., :1])], axis=-1),
    ).astype(cp.float32)

    # determine the xyz coordinates corresponding to each pixel in the texture using the weights from above
    # output is shape (texture_size, texture_size, 3) and will contain NaNs for pixels not in a triangle
    xyz_coords = cp.einsum("ijkl,ijk->ijl", vertices[triangles][triangle_numbers].astype(cp.float32), weights)
    if verbose:
        print(
            f"Got XYZ coords, shape: {xyz_coords.shape}, dtype: {xyz_coords.dtype}, time: {time.perf_counter() - start_xyz:.4f} seconds"
        )

    # for each xyz coord, find the nearest k neighbors in the original point cloud and average their colors
    start_knn = time.perf_counter()
    distances, idx = KNN_lookup_gpu(
        original_points, xyz_coords[valid_px_mask].reshape(-1, 3), nearest_neighbors
    )  # returned distances are squared
    # calculate the weighted average of the colors of the nearest neighbors
    knn_weights = 1 / (distances + 1e-8)  # in case of divide by zero
    knn_weights /= cp.sum(knn_weights, axis=1, keepdims=True)
    colors = (cp.sum(original_colors[idx] * knn_weights[..., cp.newaxis], axis=1) * 255).astype(cp.uint8)
    if verbose:
        print(
            f"Got colors, shape: {colors.shape}, dtype: {colors.dtype} time: {time.perf_counter() - start_knn:.4f} seconds"
        )

    # create texture from colors, filling in NaNs with black
    start_texture = time.perf_counter()
    texture = cp.zeros((texture_size, texture_size, 3), dtype=cp.uint8)
    texture[valid_px_mask] = colors
    # texture[triangle_numbers % 2 == 1] = [255, 0, 0] # red for upper right triangles
    if lut is not None:
        texture_lut = np.array(apply_lut_to_texture(texture, lut))
        texture = cp.array(texture_lut, dtype=cp.uint8)

    if attach_texture_to_mesh or (save_texture and output_texture_path is not None):
        texture_cpu = o3d.geometry.Image(texture.get())
    if attach_texture_to_mesh:
        mesh.textures = [texture_cpu]

    # flip the texture vertically for correct output
    texture = cp.flip(texture, axis=0)

    # create texture coords for the mesh and save output to obj file
    if generate_texture_coords:
        texture_coords = triangle_uv_coords / texture_size
        # change from row, column to x, y
        texture_coords = texture_coords.reshape(-1, 2)[:, ::-1]
        # save both to the mesh
        texture_coords = texture_coords.astype(cp.float64).get()  # float64 is faster for some reason
        mesh.triangle_uvs = o3d.utility.Vector2dVector(texture_coords)
        if verbose:
            print(
                f"Got texture coords, shape: {texture_coords.shape}, time: {time.perf_counter() - start_texture:.4f} seconds"
            )

    # attach triangles regardless of whether texture coords are generated
    mesh.triangles = o3d.utility.Vector3iVector(triangles.get())

    # save the mesh to a file
    if save_obj and output_obj_path is not None:
        start_writing_obj = time.perf_counter()
        os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
        # don't write triangle uvs to obj; will reconstruct on client side
        o3d.io.write_triangle_mesh(output_obj_path, mesh, write_triangle_uvs=include_texcoords_in_obj)
        # save uvs to separate txt file for inspection if not included in obj
        if not include_texcoords_in_obj and generate_texture_coords:
            with open(output_obj_path.replace(".obj", "_uvs.txt"), "w") as f:
                f.write("\n".join([f"{uv[0]:.6f} {uv[1]:.6f}" for uv in texture_coords]))
        if verbose:
            print(f"Finished writing obj, time: {time.perf_counter() - start_writing_obj:.4f} seconds")

    # save the texture to a file
    if save_texture and output_texture_path is not None:
        start_writing_texture = time.perf_counter()
        os.makedirs(os.path.dirname(output_texture_path), exist_ok=True)
        o3d.io.write_image(output_texture_path, texture_cpu)
        if verbose:
            print(f"Finished writing texture, time: {time.perf_counter() - start_writing_texture:.4f} seconds")

    # calculate how representative the texture is of the original point cloud
    if verbose:
        triangle_texture_sizes = (
            cp.hstack(
                [
                    [(size**2 // 2) / (texture_size * texture_size)] * number
                    for size, number in zip(triangle_sizes_list, n_triangles_list)
                ]
            )
            * 100
        )
        triangle_world_sizes = cp.hstack(split_areas).astype(cp.float64)
        triangle_world_sizes /= cp.sum(triangle_world_sizes) / 100
        representativeness = 1 / cp.linalg.norm(triangle_texture_sizes - triangle_world_sizes)  # bigger is better
        print(
            f"Texture representativeness with {texture_divisions}x{texture_divisions} grid: {representativeness:.4f}"
        )

    if verbose:
        print(f"Took {time.perf_counter() - start_time:.2f} seconds to generate the texture.")

    return mesh, texture, n_triangles_list

def calculate_max_triangles_per_bin(texture_size: int, texture_divisions: int, min_triangle_width: int = 8) -> int:
    """Calculate the maximum number of triangles that can fit in each bin of the texture"""
    # Each bin is a square of side length bin_texture_size
    bin_texture_size = texture_size // texture_divisions
    # Given a minimum triangle width, calculate the largest grid that can fit in the bin
    grid_size = bin_texture_size // min_triangle_width
    # Each cell of the grid can hold 2 triangles, lower left and upper right
    max_per_bin = (grid_size**2) * 2
    return max_per_bin



if __name__ == "__main__":
    # input_pcl = "../MeshGeneration/BlenderMeshGeneration/Meshes/Soccer/point_cloud_000000.ply"
    input_pcl = "../MeshGeneration/BlenderMeshGeneration/Meshes/Original/point_cloud_000000.ply"
    # input_mesh = "results/BlenderResult2.ply"
    # output_mesh_path = "results/BlenderResult2_with_texture.obj"
    # output_texture_path = "results/texture.png"

    input_mesh = "../MeshGeneration/BlenderMeshGeneration/generated_mesh_new.ply"
    output_mesh_path = "results/InsideMeshRemoved_with_texture_variable_8x8_k=2_cleaned_verts_py311.obj"
    output_texture_path = "results/texture_inside_removed_variable_8x8_k=2_cleaned_verts_py311.png"

    texture_size = 2048
    texture_divisions = 8  # texture wil be divided into texture_divisions x texture_divisions squares
    nearest_neighbors = 2
    buffer_pct = 25  # buffer percentage for texture coordinates

    verbose = True
    lut = None  # load_cube_file("../CanonLogToRec709.cube")

    # input_pcl = "cube_point_cloud.ply"
    # input_mesh = "test_diamond.ply"
    # output_mesh_path = "results/test_diamond_with_texture.obj"
    # output_texture_path = "results/test_diamond_texture.png"

    mesh, texture, n_triangles_list = generate_texture(
        input_pcl,
        input_mesh,
        save_obj=True,
        output_obj_path=output_mesh_path,
        save_texture=True,
        output_texture_path=output_texture_path,
        texture_size=texture_size,
        texture_divisions=texture_divisions,
        lut=lut,
        nearest_neighbors=nearest_neighbors,
        buffer_pct=buffer_pct,
        verbose=True,
    )