import os
import PIL
import time
from functools import cmp_to_key

import cupy as cp
import numpy as np
import open3d as o3d
import PyDraco
from cupy_knn import LBVHIndex
from sklearn.neighbors import KDTree
from pillow_lut import load_cube_file


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


def generate_triangle_numbers(triangle_size, grid_size, texture_size, n_triangles):
    # Generate a grid of i and j values
    i, j = cp.indices((texture_size, texture_size), dtype=cp.int32)

    # Calculate the grid row and column for each pixel
    grid_row = i // triangle_size
    grid_col = j // triangle_size

    # Calculate the triangle number for each pixel
    triangle_num = grid_row * 2 * grid_size + grid_col * 2

    # Add 1 for pixels in the upper right triangle
    triangle_num[i % triangle_size < j % triangle_size] += 1

    # Set pixels out of bounds or in excess triangles to -1
    triangle_num[
        (i >= triangle_size * grid_size) | (j >= triangle_size * grid_size) | (triangle_num >= n_triangles)
    ] = -1

    return triangle_num


def calculate_max_triangles_per_bin(texture_size: int, texture_divisions: int, min_triangle_width: int = 8) -> int:
    """Calculate the maximum number of triangles that can fit in each bin of the texture"""
    # Each bin is a square of side length bin_texture_size
    bin_texture_size = texture_size // texture_divisions
    # Given a minimum triangle width, calculate the largest grid that can fit in the bin
    grid_size = bin_texture_size // min_triangle_width
    # Each cell of the grid can hold 2 triangles, lower left and upper right
    max_per_bin = (grid_size**2) * 2
    return max_per_bin


def generate_triangle_numbers_grid(
    n_triangles_list: list[int], texture_size: int, texture_divisions: int, verbose: bool = False
):
    # preallocate arrays for overall triangle numbers and valid pixel mask
    triangle_numbers = cp.full((texture_size, texture_size), -1, dtype=cp.int32)
    valid_px_mask = cp.zeros((texture_size, texture_size), dtype=bool)
    bin_texture_size = texture_size // texture_divisions
    triangle_sizes_list = []
    grid_sizes_list = []
    for i in range(len(n_triangles_list)):
        this_indices = cp.arange(sum(n_triangles_list[:i]), sum(n_triangles_list[: i + 1]))
        this_grid_size = max(int(np.ceil(np.sqrt(n_triangles_list[i] / 2))), 1)
        this_triangle_size = bin_texture_size // this_grid_size
        if verbose:
            print(
                f"Bin {i} has {n_triangles_list[i]} triangles, grid size: {this_grid_size}, triangle size: {this_triangle_size}, "
                + f"pixels per triangle: {this_triangle_size**2 // 2}"
            )
        # record sizes for uv coords later
        grid_sizes_list.append(this_grid_size)
        triangle_sizes_list.append(this_triangle_size)
        # generate triangle numbers for this portion of the texture with indices 0 - n_triangles_list[i]
        this_triangle_numbers = generate_triangle_numbers(
            this_triangle_size, this_grid_size, bin_texture_size, n_triangles_list[i]
        )
        this_valid_px_mask = this_triangle_numbers > -1
        # replace with the original indices from the sorted list
        this_triangle_numbers[this_valid_px_mask] = this_indices[this_triangle_numbers[this_valid_px_mask]]
        triangle_numbers[
            (i // texture_divisions) * bin_texture_size : (i // texture_divisions + 1) * bin_texture_size,
            (i % texture_divisions) * bin_texture_size : (i % texture_divisions + 1) * bin_texture_size,
        ] = this_triangle_numbers
        valid_px_mask[
            (i // texture_divisions) * bin_texture_size : (i // texture_divisions + 1) * bin_texture_size,
            (i % texture_divisions) * bin_texture_size : (i % texture_divisions + 1) * bin_texture_size,
        ] = this_valid_px_mask
    return triangle_numbers, valid_px_mask, triangle_sizes_list, grid_sizes_list


def generate_triangle_uv_vertex_coords(n_triangles: int, triangle_size: int, grid_size: int) -> cp.ndarray:
    # Create an array to hold the UV coordinates
    uv_coords = cp.empty((n_triangles, 3, 2), dtype=cp.float32)

    # Calculate the grid row and column for each triangle
    triangle_nums = cp.arange(n_triangles)
    grid_rows = triangle_nums // (2 * grid_size)
    grid_cols = (triangle_nums % (2 * grid_size)) // 2
    is_upper_right = triangle_nums % 2 == 1

    # Calculate the UV coordinates for the upper right triangles
    upper_right_row_px = grid_rows[is_upper_right] * triangle_size
    upper_right_col_px = grid_cols[is_upper_right] * triangle_size
    uv_coords[is_upper_right, 0, :] = cp.stack(
        [upper_right_row_px, upper_right_col_px + triangle_size - 1], axis=-1
    )  # top right
    uv_coords[is_upper_right, 1, :] = cp.stack([upper_right_row_px, upper_right_col_px + 1], axis=-1)  # top left
    uv_coords[is_upper_right, 2, :] = cp.stack(
        [upper_right_row_px + triangle_size - 2, upper_right_col_px + triangle_size - 1],
        axis=-1,
    )  # bottom right

    # Calculate the UV coordinates for the lower left triangles
    lower_left_row_px = grid_rows[~is_upper_right] * triangle_size
    lower_left_col_px = grid_cols[~is_upper_right] * triangle_size
    uv_coords[~is_upper_right, 0, :] = cp.stack(
        [lower_left_row_px + triangle_size - 1, lower_left_col_px + triangle_size - 1],
        axis=-1,
    )  # bottom right
    uv_coords[~is_upper_right, 1, :] = cp.stack([lower_left_row_px, lower_left_col_px], axis=-1)  # top left
    uv_coords[~is_upper_right, 2, :] = cp.stack(
        [lower_left_row_px + triangle_size - 1, lower_left_col_px], axis=-1
    )  # bottom left

    return uv_coords


def generate_triangle_uv_vertex_coords_grid(
    n_triangles_list: list[int],
    triangle_sizes_list: list[int],
    grid_sizes_list: list[int],
    texture_size: int,
    texture_divisions: int,
):
    # preallocate array for all uv coords
    n_triangles = sum(n_triangles_list)
    bin_texture_size = texture_size // texture_divisions
    triangle_uv_coords = cp.empty((n_triangles, 3, 2), dtype=cp.float32)
    for i, (this_n_triangles, this_triangle_size, this_grid_size) in enumerate(
        zip(n_triangles_list, triangle_sizes_list, grid_sizes_list)
    ):
        this_split_indices = cp.arange(sum(n_triangles_list[:i]), sum(n_triangles_list[: i + 1]))
        this_triangle_uv_coords = generate_triangle_uv_vertex_coords(
            this_n_triangles, this_triangle_size, this_grid_size
        )  # shape (this_n_triangles, 3, 2)
        # offset by the correct amount for this portion of the texture
        this_triangle_uv_coords += cp.array(
            [(i // texture_divisions) * bin_texture_size, (i % texture_divisions) * bin_texture_size], dtype=cp.float32
        )[None, None, :]
        triangle_uv_coords[this_split_indices] = this_triangle_uv_coords
    return triangle_uv_coords


def buffer_uv_coords_barycenter(uv_coords: cp.array, buffer_pct: float) -> cp.array:
    # Calculate the barycenter of each triangle; uv_coords is shape (n_triangles, 3, 2)
    barycenters = cp.mean(uv_coords, axis=1, keepdims=True)
    # Compute a vector from each vertex to the barycenter and scale it by the buffer percentage
    distances = barycenters - uv_coords
    buffer = distances * buffer_pct / 100
    # Output is original uv_coords with each vertex moved toward the barycenter by the buffer amount
    return uv_coords + buffer


def quantize_vertices(
    vertices: cp.ndarray,
    n_bits: int,
    min_vals: cp.ndarray = None,
    max_vals: cp.ndarray = None,
    return_int: bool = False,
) -> cp.ndarray:
    if n_bits < 1 or n_bits > 32:
        raise ValueError("n_bits must be between 1 and 32.")

    # first quantize to integers
    min_vals = cp.min(vertices, axis=0) if min_vals is None else min_vals
    max_vals = cp.max(vertices, axis=0) if max_vals is None else max_vals
    new_vertices = (vertices - min_vals) / (max_vals - min_vals)
    new_vertices = (new_vertices * (2**n_bits - 1)).astype(cp.int32)
    # return the integer values if requested
    if return_int:
        return new_vertices

    # now convert back to float
    new_vertices = (new_vertices / (2**n_bits - 1) * (max_vals - min_vals) + min_vals).astype(cp.float32)
    return new_vertices


def quantize_vertices_draco(
    faces: np.ndarray, vertices: np.ndarray, tex_coords: np.ndarray, n_bits: int, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if n_bits < 1 or n_bits > 32:
        raise ValueError("n_bits must be between 1 and 32.")

    start_encode = time.perf_counter()
    encoded = PyDraco.encode(
        faces=faces.flatten(),
        vertices=vertices.flatten(),
        # tex_coords=tex_coords.flatten() if tex_coords is not None else None,
        compression_level=6,
        # pos_quantization_bits=n_bits,
        # generic_quantization_bits=8,
        # tex_coords_quantization_bits=9,
    )
    if verbose:
        print(f"Encoded size: {len(encoded)} bytes")
    # with open("draco_encoded_int.drc", "wb") as f:
    #     f.write(encoded)

    if verbose:
        print(f"Quantized vertices, time: {time.perf_counter() - start_encode:.4f} seconds")

    start_decode = time.perf_counter()
    decoded = PyDraco.decode(encoded)
    if verbose:
        print(f"Decoded vertices, time: {time.perf_counter() - start_decode:.4f} seconds")
    return (
        decoded.vertices.reshape(-1, 3),
        decoded.faces.reshape(-1, 3),
        decoded.tex_coords.reshape(-1, 2) if len(decoded.tex_coords) > 0 else decoded.tex_coords,
        (
            decoded.tex_coords_indices.reshape(-1, 3)
            if len(decoded.tex_coords_indices) > 0
            else decoded.tex_coords_indices
        ),
    )


def bin_xyz_coords(xyz: cp.array, voxel_size: float | list[float]) -> cp.array:
    """Bin the xyz coordinates into voxels of the specified size and return a linear index for each point"""
    volume_limits = cp.array(
        [
            [-16000, 16000],  # x limits
            [-8500, 8500],  # y limits
            [0, 5500],  # z limits
        ]
    )
    voxel_shape = ((volume_limits[:, 1] - volume_limits[:, 0]) // voxel_size).astype(cp.int32)
    if voxel_shape[0] * voxel_shape[1] * voxel_shape[2] >= 2**31:
        raise ValueError("Voxel grid is too large, increase voxel size.")

    if ~isinstance(voxel_size, list):
        voxel_size = [voxel_size, voxel_size, voxel_size]
    elif len(voxel_size) != 3:
        raise ValueError("Voxel size must be a float or a list of 3 floats.")
    voxel_size = cp.array(voxel_size)

    # first clip points to volume limits
    xyz = cp.clip(xyz, volume_limits[:, 0], volume_limits[:, 1])

    # Calculate the voxel indices for each point
    voxel_indices = cp.floor(xyz / voxel_size).astype(cp.int32)
    voxel_indices -= (volume_limits[:, 0] // voxel_size).astype(cp.int32)

    # Calculate the linear index for each point
    linear_indices = (
        voxel_indices[:, 0] * voxel_shape[1] * voxel_shape[2]
        + voxel_indices[:, 1] * voxel_shape[2]
        + voxel_indices[:, 2]
    )
    return linear_indices


def KNN_lookup_binned(
    original_points: cp.ndarray, xyz_coords: cp.ndarray, nearest_neighbors: int, bin_size: float
) -> tuple[cp.ndarray, cp.ndarray]:
    # bin the original xyz coordinates into voxels for faster nearest neighbor search
    original_points_bin_indices = bin_xyz_coords(original_points, voxel_size=bin_size)
    xyz_coords_bin_indices = bin_xyz_coords(xyz_coords, voxel_size=bin_size)
    distances = cp.empty((len(xyz_coords), nearest_neighbors), dtype=cp.float32)
    idx = cp.empty((len(xyz_coords), nearest_neighbors), dtype=cp.int32)
    unique_bins, bin_ct = cp.unique(xyz_coords_bin_indices, return_counts=True)
    print(
        f"Found {len(unique_bins)} unique bins of side length {bin_size} mm, max bin count: {bin_ct.max()}, "
        + f"min bin count: {bin_ct.min()}, avg bin count: {bin_ct.mean():.2f}"
    )
    # for each bin, find the nearest neighbors in the original point cloud, looking only at nearby bins
    nearby_pts_ct = []
    small_bins_mask = cp.zeros(len(xyz_coords_bin_indices), dtype=bool)
    nearby_bin_shifts = cp.array(
        [[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]], dtype=cp.float32
    )
    for bin_num in unique_bins:
        bin_mask = xyz_coords_bin_indices == bin_num
        this_bin_pts = xyz_coords[bin_mask]
        this_nearby_bins = bin_xyz_coords(
            this_bin_pts[0:1] + nearby_bin_shifts * bin_size,
            voxel_size=bin_size,
        )
        nearby_mask = cp.isin(original_points_bin_indices, this_nearby_bins)
        nearby_pts = original_points[nearby_mask]
        nearby_indices = cp.where(nearby_mask)[0]
        nearby_pts_ct.append(len(nearby_pts))
        # if a bin has fewer than nearest_neighbors points, cannot find that many nearest neighbors
        if len(nearby_pts) < nearest_neighbors:
            small_bins_mask[bin_mask] = True
            continue
        # print(f"Bin {bin_num} has {len(this_bin_pts)} points and nearby bins {this_nearby_bins} which contain {len(nearby_pts)} points.")
        kdtree = KDTree(nearby_pts.get())
        this_distances, this_idx = kdtree.query(this_bin_pts.get(), k=nearest_neighbors)
        distances[bin_mask] = this_distances
        idx[bin_mask] = nearby_indices[this_idx]

    # for small bins, find the nearest neighbors in the original point cloud, looking at all points
    if cp.any(small_bins_mask):
        print(f"Found {small_bins_mask.sum()} points in small bins, searching through all points for these")
        small_bin_pts = xyz_coords[small_bins_mask]
        kdtree = KDTree(original_points.get())
        this_distances, this_idx = kdtree.query(small_bin_pts.get(), k=nearest_neighbors)
        distances[small_bins_mask] = this_distances
        idx[small_bins_mask] = this_idx

    print(
        f"Nearby points count: {nearby_pts_ct}, max: {max(nearby_pts_ct)}, min: {min(nearby_pts_ct)}, "
        + f"avg: {sum(nearby_pts_ct) / len(nearby_pts_ct):.2f}"
    )

    return distances, idx


def KNN_lookup_gpu(
    original_points: cp.ndarray, xyz_coords: cp.ndarray, nearest_neighbors: int
) -> tuple[cp.ndarray, cp.ndarray]:
    lbvh = LBVHIndex()
    lbvh.build(original_points)
    lbvh.prepare_knn_default(nearest_neighbors)
    idx, distances, _ = lbvh.query_knn(xyz_coords, nearest_neighbors)
    return distances, idx


def triangles_to_unique_ids(triangles: cp.ndarray, vertices: cp.ndarray) -> cp.ndarray:
    """
    Given a list of triangles with indices [a, b, c] and a list of vertices, return a list of unique IDs for each triangle.
    The unique ID is based on the vertices of the triangle and should not be sensitive to small changes in vertex coordinates.
    """
    # Get the coordinates of the vertices of each triangle
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    triangle_vertices = cp.concatenate([v0, v1, v2], axis=1)  # now shape (n_triangles, 9)

    # We know the maximium/minimum values of the coordinates, so we can quantize them to a fixed number of bits
    triangle_vertices = quantize_vertices(
        triangle_vertices,
        n_bits=16,
        min_vals=cp.array([-16000, -8500, 0] * 3),  # 9 coords since we concatenated the vertices
        max_vals=cp.array([16000, 8500, 5500] * 3),
        return_int=True,
    )

    # Sort the vertices of each triangle so that the same triangle will always have the same sorted vertices;
    # this may not be necessary if draco does not reorder vertices within triangles
    sorted_triangle_vertices = cp.sort(triangle_vertices, axis=1)

    # Calculate a unique ID for each triangle based on the sorted vertices; since each coordinate can be
    return None


def fuzzy_compare(a, b, eps, reverse=False):
    """
    Allows sorting with a tolerance for floating point errors. Returns 0 if a and b are within eps, -1 if a < b, 1 if a > b.
    If reverse is True, the comparison is reversed.
    """
    if abs(a - b) < eps:
        return 0
    else:
        if reverse:
            return -1 if a > b else 1
        else:
            return -1 if a < b else 1


def fuzzy_argsort(data: cp.ndarray, eps: float, reverse: bool = False) -> cp.ndarray:
    """Sorts data with a tolerance for floating point errors. Returns the indices that would sort the data."""
    enumerated_data = list(enumerate(data.get()))  # Pair each element with its original index
    enumerated_data.sort(
        key=cmp_to_key(lambda x, y: fuzzy_compare(x[1], y[1], eps, reverse=reverse))
    )  # Sort with tolerance
    sorted_indices = cp.array([x[0] for x in enumerated_data], dtype=cp.int32)  # Extract the original indices
    return sorted_indices


def calculate_and_sort_triangle_sizes(
    triangles: cp.ndarray, vertices: cp.ndarray, secondary_sort: str = None, argsort_eps: float = 0.01, **kwargs
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Given a (n_triangles, 3) array of vertex indices and a (n_vertices, 3) array of vertex coordinates, calculate the
    area of each triangle and sort the triangles by area, largest first.

    Also allows secondary sort of "sum" to sort by the sum of the coordinates of the vertices, or "bin" to sort by bin.
    If "bin" is chosen, the voxel_size (in mm) may be passed as a keyword argument, e.g. voxel_size=100 for 10 cm bins.

    Returns the sorted indices and sizes of the triangles.
    """
    # Calculate the area of each triangle
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    a = cp.linalg.norm(cp.cross(v1 - v0, v2 - v0), axis=1) / 2
    # # print(a[:10], cp.round(a,0)[:10])
    # # shift up so that smallest area is 1e12
    # a = cp.round(a, 0).astype(cp.uint64) * 10**12
    # # print(a[:10])
    # # use the 9 digits to encode xyz coords of the center of the triangle
    # triangle_centers = (v0 + v1 + v2) / 3
    # # print(triangle_centers[:10])
    # hash_val = (cp.abs(triangle_centers) * 1e4 / cp.abs(triangle_centers).max()).astype(cp.uint64) # each xyz is now 4 digits
    # # print(hash_val[:10])
    # hash_val = cp.sum(hash_val * cp.array([10**8, 10**4, 1], dtype=cp.uint64), axis=1)  # combine xyz into one number
    # # print(hash_val[:10])
    # a += hash_val  # add the hash to the area to break ties
    # # print(a[:10])

    if secondary_sort == "sum":
        # modified sort
        modifed_a = a**2 * 400000 + cp.sum(v0, axis=1) + cp.sum(v1, axis=1) + cp.sum(v2, axis=1)
        intermediate_sort = cp.argsort(modifed_a)
        a_int = a[intermediate_sort]
    elif secondary_sort == "dumb":
        a_rounded = cp.round(a, 1)  # round to 1 decimal place to force more ties in area
        triangle_centers = (v0 + v1 + v2) / 3
        dumb_val = triangle_centers[:, 0] * 1e12 + triangle_centers[:, 1] * 1e6 + triangle_centers[:, 2]
        intermediate_sort = cp.argsort(dumb_val)
        a_int = a_rounded[intermediate_sort]
    elif secondary_sort == "bin":
        # sort first by bin, then by area
        voxel_size = kwargs.get("voxel_size", 100)  # default to 100 mm = 10 cm
        triangle_centers = (v0 + v1 + v2) / 3
        triangle_bins = bin_xyz_coords(triangle_centers, voxel_size)
        intermediate_sort = cp.argsort(triangle_bins)
        a_int = a[intermediate_sort]
    else:
        # if no intermediate sort, just sort by area
        intermediate_sort = cp.arange(len(a), dtype=cp.int32)
        a_int = a.copy()

    # Sort the triangles by area, largest first; use custon comparator to allow secondary sort to take over
    # even when there are not EXACT ties in area
    # sorted_indices = fuzzy_argsort(a_int, argsort_eps, reverse=True)
    sorted_indices = cp.argsort(a_int)[::-1]
    # go back to original order if there was an intermediate sort
    sorted_indices = intermediate_sort[sorted_indices]
    sorted_sizes = a[sorted_indices]
    return sorted_indices, sorted_sizes


def calculate_and_sort_triangle_sizes_int(
    triangles: cp.ndarray, vertices: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Given a (n_triangles, 3) array of vertex indices and a (n_vertices, 3) array of vertex coordinates, calculate the
    squared area of each triangle and sort the triangles by that, largest first.

    Ties are broken by sorting all 9 coordinates of the triangle vertices, then using these sorted 9 values as secondary sorts.

    Returns the sorted indices and sizes of the triangles.
    """
    # Calculate the area of each triangle
    v0 = vertices[triangles[:, 0]].astype(cp.int64)  # convert to int64 to prevent overflow
    v1 = vertices[triangles[:, 1]].astype(cp.int64)
    v2 = vertices[triangles[:, 2]].astype(cp.int64)
    a_squared = cp.sum(cp.cross(v1 - v0, v2 - v0) ** 2, axis=1)

    x_sorted = cp.sort(cp.vstack((v0[:, 0], v1[:, 0], v2[:, 0])).T, axis=1)
    y_sorted = cp.sort(cp.vstack((v0[:, 1], v1[:, 1], v2[:, 1])).T, axis=1)
    z_sorted = cp.sort(cp.vstack((v0[:, 2], v1[:, 2], v2[:, 2])).T, axis=1)
    to_sort = cp.concatenate((z_sorted, y_sorted, x_sorted), axis=1)
    to_sort = cp.concatenate((to_sort, a_squared[:, cp.newaxis]), axis=1)
    sorted_indices = cp.lexsort(to_sort.T)[::-1]
    sorted_sizes = cp.sqrt(a_squared[sorted_indices]) / 2  # return the actual area, not the squared area
    return sorted_indices, sorted_sizes


def split_triangles_into_equal_area_bins(
    triangles: cp.ndarray,
    vertices: cp.ndarray,
    n_bins: int = 4,
    max_per_bin: int = cp.inf,
    return_sizes: bool = False,
    remove_small: bool = False,
    area_threshold_percent: float = 0.1,
) -> list[cp.ndarray]:
    if len(triangles) > n_bins * max_per_bin:
        # too many triangles
        raise ValueError(
            f"Cannot split {len(triangles)} triangles into {n_bins} bins of max {max_per_bin} triangles each."
        )

    area_sorted_indices, area_sorted_sizes = calculate_and_sort_triangle_sizes_int(triangles, vertices)

    # remove small triangles smaller than area_threshold percent of the mean triangle area
    if remove_small:
        small_indices = area_sorted_sizes < area_threshold_percent / 100 * cp.mean(area_sorted_sizes)
        area_sorted_indices = area_sorted_indices[~small_indices]
        area_sorted_sizes = area_sorted_sizes[~small_indices]

    # Divide the triangles into n_bins equal area bins using cumulative sum; first bin contains largest 1/n_bins triangles, etc.
    area_cumsum = cp.cumsum(area_sorted_sizes)
    area_cumsum /= area_cumsum[-1]
    area_bin_indices = cp.searchsorted(area_cumsum, cp.arange(1, n_bins) / n_bins)
    # validate split to ensure no bin has more than max_per_bin triangles
    min_indices = len(area_sorted_sizes) - cp.arange(n_bins - 1, 0, -1) * max_per_bin
    area_bin_indices = cp.maximum(min_indices, area_bin_indices).get().tolist()  # must be list on CPU for split

    # split triangles into bins
    split_indices = cp.split(area_sorted_indices, area_bin_indices)
    if return_sizes:
        return split_indices, cp.split(area_sorted_sizes, area_bin_indices)
    return split_indices


def sort_triangle_vertices(triangles: cp.ndarray, vertices: cp.ndarray) -> cp.ndarray:
    triangle_vertices = vertices[triangles]  # shape (n_triangles, 3, 3)
    # sort the vertices within each triangle by x first, then y, then z
    original_order = cp.repeat(cp.arange(3, dtype=cp.int32).reshape(1, -1), len(triangles), axis=0)  # shape (n, 3)
    idx = cp.arange(len(triangles))[:, None]
    for i in range(3):
        sorted_order = cp.argsort(triangle_vertices[:, :, i], axis=1)  # shape (n_triangles, 3)
        original_order = original_order[idx, sorted_order]
        triangle_vertices = triangle_vertices[idx, sorted_order]
    # want to keep the normals correct, so only use the first vertex, then make the others follow their original order
    original_order_normals_preserved = cp.vstack(
        [original_order[:, 0], (original_order[:, 0] + 1) % 3, (original_order[:, 0] + 2) % 3]
    ).T
    sorted_triangles = triangles[idx, original_order_normals_preserved]
    return sorted_triangles


def remove_permuted_faces(triangles: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Given an array of triangle faces shape (n, 3), remove any faces that are odd permutations of another face.
    The faces are assumed to have their vertices already sorted so that the first vertex is fixed.
    """
    triangles_cpu = [tuple(x) for x in triangles.get().tolist()]  # convert to tuple for hashability
    triangles_set = set(triangles_cpu)  # set allows O(1) in operator
    removed_indices = [
        i
        for i in range(len(triangles_cpu))
        if (triangles_cpu[i][0], triangles_cpu[i][2], triangles_cpu[i][1]) in triangles_set
    ]
    triangles_cpu = np.delete(triangles_cpu, removed_indices, axis=0)
    return cp.array(triangles_cpu), cp.array(removed_indices)


def apply_lut_to_texture(texture: cp.ndarray, lut) -> PIL.Image:
    # Apply the LUT to the texture
    texture = cp.asnumpy(texture)
    texture = PIL.Image.fromarray(texture)
    texture = texture.filter(lut)
    return texture


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
