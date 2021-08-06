import numpy as np
from initializer import Initializer
from visualizer import Visualizer
import open3d as o3d


def generate_header(num_points, with_normals):
    # Generate a header for the .ply file with or without normals
    if with_normals:
        header = ["ply\n",
                  "format ascii 1.0\n",
                  f'element vertex {num_points}\n',
                  "property float x\n",
                  "property float y\n",
                  "property float z\n",
                  "property float nx\n",
                  "property float ny\n",
                  "property float nz\n",
                  "property uchar red\n",
                  "property uchar green\n",
                  "property uchar blue\n",
                  "end_header\n"]
    else:
        header = ["ply\n",
                  "format ascii 1.0\n",
                  f'element vertex {num_points}\n',
                  "property float x\n",
                  "property float y\n",
                  "property float z\n",
                  "property uchar red\n",
                  "property uchar green\n",
                  "property uchar blue\n",
                  "end_header\n"]
    return header


def calculate_view_directions(views):
    """
    Creates an array of vectors with the direction of the optical axis
    ------
    Input:
        views : A list of views (book_keeper.Q)
    Output:
        view_directions : # 3x36 numpy array
    """
    view_directions = np.empty([3, 0])
    for view in views:
        view_direction = view.rotation_matrix.T @ np.array([[0, 0, 1]]).T
        view_directions = np.hstack([view_directions, view_direction]) if view_directions.size else view_direction
    return view_directions


def get_camera_indices(book_keeper, point_idx):
    """
    Returns indices of cameras that can see the 3D point
    """
    return np.nonzero(book_keeper.vis_matrix[:, point_idx])[0]  # vis_matrix 36x676


def generate_mesh(book_keeper):
    """
    Generates mesh from point cloud
    """
    point_cloud = book_keeper.P.T  # Extract point cloud from book_keeper
    vis = Visualizer()
    estimated_colors = vis.estimate_color(book_keeper.Q)  # Estimate colors of 3D points

    with open("images/dino_cloud_without_normals.ply", "w") as ply_file:
        header = generate_header(point_cloud.shape[0], with_normals=False)
        ply_file.writelines(header)
        for point, estimated_color in zip(point_cloud, estimated_colors):
            # x, y, z, r, g, b
            ply_file.write(
                f'{point[0]} {point[1]} {point[2]} {round(estimated_color[0] * 255)} {round(estimated_color[1] * 255)} {round(estimated_color[2] * 255)}\n')

    # Using open3d to estimate normals to the 3D points
    o3d_point_cloud = o3d.io.read_point_cloud("images/dino_cloud_without_normals.ply")
    o3d_point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(o3d_point_cloud.normals)  # Nx3

    view_directions = calculate_view_directions(book_keeper.Q)  # 3x36
    new_normals = np.ones(normals.shape)

    # Refine the normals
    for point, index_3d_2d, point_idx in zip(point_cloud, book_keeper.index_3d_2d, range(point_cloud.shape[0])):
        camera_indices = get_camera_indices(book_keeper, index_3d_2d)  # Get indices of cameras that can see 3D point
        num_true = 0  # How many cameras agree with the current normal
        for camera_idx in camera_indices:
            # Check if normal has correct sign
            if view_directions[:, camera_idx] @ normals[point_idx].reshape(3, 1) < 0:
                num_true += 1
        if camera_indices.size > 0:
            # Flip normal if less than 50% of the cameras agree with the current direction
            if num_true / camera_indices.size < 0.5:
                new_normals[point_idx] = -normals[point_idx]
            else:
                new_normals[point_idx] = normals[point_idx]

    with open("images/dino_cloud_with_normals.ply", "w") as ply_file:
        normal_header = generate_header(point_cloud.shape[0], with_normals=True)
        ply_file.writelines(normal_header)
        for point, normal, estimated_color in zip(point_cloud, new_normals, estimated_colors):
            # x, y, z, nx, ny, nz, r, g, b
            ply_file.write(
                f'{point[0]} {point[1]} {point[2]} {normal[0]} {normal[1]} {normal[2]} {round(estimated_color[0] * 255)} {round(estimated_color[1] * 255)} {round(estimated_color[2] * 255)}\n')
