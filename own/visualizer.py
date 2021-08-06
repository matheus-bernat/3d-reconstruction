#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from external import lab3
import warnings

class Visualizer():
    """
    Declares functions used for visualization.
    """
    def __init__(self):
        pass

    def visualize_corresp(self, view_1, view_2, p_index):
        """
        Displays the correspondences between the 2 given View objects.
        """
        proj_1 = view_1.projections[:, p_index]
        proj_2 = view_2.projections[:, p_index]
        lab3.show_corresp(view_1.image, view_2.image, proj_1, proj_2, vertical=False)
        plt.title(f'Cameras {view_1.id} ({view_1.camera_center[0,0]:.0f}, {view_1.camera_center[1,0]:.0f}, {view_1.camera_center[2,0]:.0f})'
                  f' and {view_2.id} ({view_2.camera_center[0,0]:.0f}, {view_2.camera_center[1,0]:.0f}, {view_2.camera_center[2,0]:.0f})')
        #plt.show()

    def visualize_3d_points(self, book_keeper, m, estimate_color=True):
        """
        Displays a point cloud.
        ---
        Input:
        point_cloud : [int, int]
            [Nx3] matrix representing the 3D points in the point cloud
            (N = number of 3D points)
        """
        if estimate_color:
            est_col = self.estimate_color(book_keeper.Q)

        z_hat = np.array([0, 0, 1]).reshape(3, 1)

        point_cloud = book_keeper.P.T  # nx3
        n = book_keeper.P.shape[1]

        # only 3d points
        fig_3d = plt.figure('3d point cloud')
        ax_3d = fig_3d.gca(projection='3d')
        if estimate_color:
            ax_3d.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                              color=est_col[book_keeper.index_3d_2d])
        else:
            ax_3d.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                              color=(0, 0, 1))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Existing 3d points
        ax.scatter(point_cloud[0:n-m, 0], point_cloud[0:n-m, 1], point_cloud[0:n-m, 2], color=(0, 0, 1))
        # Newly added 3d points
        ax.scatter(point_cloud[n-m:, 0], point_cloud[n-m:, 1], point_cloud[n-m:, 2], color=(1, 0, 1))
        for view_idx in range(len(book_keeper.Q)):
            if book_keeper.Q[view_idx].id == 0:
                # origin camera pose
                ax.scatter(book_keeper.Q[view_idx].camera_center[0], book_keeper.Q[view_idx].camera_center[1], book_keeper.Q[view_idx].camera_center[2], color=(0, 1, 0))
                ax.text(book_keeper.Q[view_idx].camera_center[0, 0], book_keeper.Q[view_idx].camera_center[1, 0],
                        book_keeper.Q[view_idx].camera_center[2, 0], f'C{book_keeper.Q[view_idx].id}')
                ax.quiver(0, 0, 0, 0, 0, 1)
            elif view_idx == len(book_keeper.Q) - 1:
                # newly added camera pose
                ax.scatter(book_keeper.Q[view_idx].camera_center[0], book_keeper.Q[view_idx].camera_center[1],
                           book_keeper.Q[view_idx].camera_center[2], color=(1, 0, 0))
                ax.text(book_keeper.Q[view_idx].camera_center[0, 0], book_keeper.Q[view_idx].camera_center[1, 0],
                        book_keeper.Q[view_idx].camera_center[2, 0], f'C{book_keeper.Q[view_idx].id}')
                view_direction = book_keeper.Q[view_idx].rotation_matrix.T @ z_hat
                ax.quiver(book_keeper.Q[view_idx].camera_center[0], book_keeper.Q[view_idx].camera_center[1],
                          book_keeper.Q[view_idx].camera_center[2], view_direction[0], view_direction[1],
                          view_direction[2])
            else:
                # already added camera poses
                ax.scatter(book_keeper.Q[view_idx].camera_center[0], book_keeper.Q[view_idx].camera_center[1], book_keeper.Q[view_idx].camera_center[2], color=(0, 0, 0))
                ax.text(book_keeper.Q[view_idx].camera_center[0, 0], book_keeper.Q[view_idx].camera_center[1, 0],
                        book_keeper.Q[view_idx].camera_center[2, 0], f'C{book_keeper.Q[view_idx].id}')
                view_direction = book_keeper.Q[view_idx].rotation_matrix.T @ z_hat
                ax.quiver(book_keeper.Q[view_idx].camera_center[0], book_keeper.Q[view_idx].camera_center[1],
                          book_keeper.Q[view_idx].camera_center[2], view_direction[0], view_direction[1],
                          view_direction[2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        var = str(len(book_keeper.Q))
        # plt.savefig('images/' + var + '_plot.png', bbox_inches='tight', dpi=250)
        # plt.close('all')
        plt.show()

    def visualize_reprojection(self, view, points_3d):
        """
        Visualize reprojection of already triangulated 3d points on top of view.image.
        ------
        Inputs:
            view : View
            points_3d : 3xN
        """
        lab3.imshow(view.image)
        reproj = lab3.project(points_3d, view.camera_matrix)
        plt.scatter(reproj[0, :], reproj[1, :], color=(1, 0, 0))
        # plt.show()


    def estimate_color(self, view):
        """
        Estimates the color for 3D points in a point cloud.
        ---
        Input
            view : [View, View]
                List of view objects
        ---
        Output:
            estimated_color :
                [Nx3] matrix representing the normalized rgb color for every 3D point
                (N = number of 3D points)
        """
        num_3d_points = view[0].projections.shape[1]
        num_cameras = len(view)
        estimated_color = np.zeros((num_3d_points, 3))

        for point_idx in range(num_3d_points):
            color_sum = np.zeros((3))
            num_views_visible = 0
            for camera_idx in range(num_cameras):
                pixel_x = round(view[camera_idx].projections[0][point_idx])
                pixel_y = round(view[camera_idx].projections[1][point_idx])
                if pixel_x != -1:
                    color_sum += view[camera_idx].image[pixel_y, pixel_x]
                    num_views_visible += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # ignore a warning with nanmean
                estimated_color[point_idx] = color_sum/(255*num_views_visible)

        return estimated_color

    def visualize_camera_centers(self, R_est, t_est, R_gt, t_gt):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        gt_cameras = np.zeros([R_gt.shape[0], 3, 1])
        est_cameras = np.zeros([R_est.shape[0], 3, 1])
        for i in range(R_gt.shape[0]):
            gt_cameras[i, ...] = -R_gt[i, ...].T @ t_gt[i, ...]
            est_cameras[i, ...] = -R_est[i, ...].T @ t_est[i, ...]
        ax.scatter(gt_cameras[:, 0], gt_cameras[:, 1], gt_cameras[:, 2], color=(1, 0, 0))
        ax.scatter(est_cameras[:, 0], est_cameras[:, 1], est_cameras[:, 2], color=(0, 0, 1))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def visualize_interest_points(self, view):
        """
        Displays the 2D projections of the given View object on the actual image.
        """
        plt.imshow(view.image)
        plt.scatter(view.projections[0, :], view.projections[1, :], c='r', label='o')
        plt.show()


