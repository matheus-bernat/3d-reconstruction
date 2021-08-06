#!/usr/bin/env python3

from external import lab3
from help_functions import is_consistent_with_F
import numpy as np
import cv2


class Assembler():
    """
    Class containing the functions necessary for the steps EXT1-EXT5 and WASH2.
    """

    def __init__(self, K):
        self.K = K

    def choose_new_view(self, all_views):
        """
        Returns an unused View object based on the list of all the views and the unused Views.
        Assumes that camera 1 must be used.
        """
        is_prev_view_used = False
        prev_view = []
        for view in all_views:
            if view.used:
                is_prev_view_used = True
                prev_view = view
            else:
                if is_prev_view_used:
                    return prev_view, view

    def get_2d_3d_corresp(self, new_view, index_3d_2d, P):
        """
        Returns a list of 2D points that has triangulated 3D points and a list of the
        corresponding 3D points.
        Returns:
            points_2d : np array 2xN
            points_3d : np array 3xN
        """
        points_2d = np.array([], dtype=np.int64).reshape(2, 0)
        points_3d = np.array([], dtype=np.int64).reshape(3, 0)
        for i in range(index_3d_2d.shape[0]):
            if new_view.projections[0, index_3d_2d[i]] != -1:
                p_2d = new_view.projections[:, index_3d_2d[i]].reshape(2, 1)
                points_2d = np.hstack([points_2d, p_2d]) if points_2d.size else p_2d
                p_3d = P[:, i].reshape(3, 1)
                points_3d = np.hstack([points_3d, p_3d]) if points_3d.size else p_3d
        return points_2d, points_3d

    def robust_pnp(self, points_2d, points_3d, max_iter=1000, threshold=10):
        """
        Robustly estimates the camera pose [R|t] given 3d points and their believed projections 2d.
        Returns R, t and the consensus set (i.e. inliers whose projection errors are within threshold).
        ------
        Input:
            points_2d : np array 2xN
            points_3d : np array 3xN
        """
        if points_2d.shape[1] < 4:
            raise ValueError('ERROR: Not enough points to estimate R and t with PnP RANSAC')
        K_no_skew = np.copy(self.K)
        K_no_skew[0, 1] = 0
        _, R_vec, t, inliers_idx = cv2.solvePnPRansac(points_3d.T, points_2d.T, K_no_skew, distCoeffs=None,
                                                      flags=cv2.SOLVEPNP_P3P, reprojectionError=threshold,
                                                      confidence=0.999, iterationsCount=max_iter)

        R, _ = cv2.Rodrigues(R_vec)
        inliers_2d = points_2d[:, inliers_idx][:, :, 0]
        inliers_3d = points_3d[:, inliers_idx][:, :, 0]

        return R, t, inliers_2d, inliers_3d

    def get_E_from_F(self, prev_view, new_view):
        """
        Determine essential matrix from both View objects' camera matrices. Ref: le 11 slide 27.
        """
        F = lab3.fmatrix_from_cameras(prev_view.camera_matrix, new_view.camera_matrix)
        E = self.K.T @ F @ self.K
        return E

    def get_F_from_E(self, E):
        """Returns the normalized fundamental matrix F given essential matrix E and intrinsics K"""
        F = np.linalg.inv(self.K.T) @ E @ np.linalg.inv(self.K)
        return F

    def add_3d_points(self, prev_view, new_view, book_keeper, F, epi_constraint_thresh):
        """
        Triangulates feasible correspondences, and add resulting 3D points to book_keeper.P.
        """
        index_3d_2d = book_keeper.index_3d_2d
        added_idx = []
        # Get which points can be triangulated: need to be visible by both views and consistent with E
        for idx, prev_point, new_point in zip(range(new_view.projections.shape[1]), prev_view.projections.T,
                                              new_view.projections.T):
            if prev_point[0] != -1 and new_point[0] != -1 and idx not in index_3d_2d:
                triangulated_point = (lab3.triangulate_optimal(prev_view.camera_matrix, new_view.camera_matrix,
                                                               prev_view.projections[:, idx],
                                                               new_view.projections[:, idx])).reshape(3, 1)
# look through indicies with triangulation
                if is_consistent_with_F(prev_point, new_point, F, epi_constraint_thresh):
                    book_keeper.add_point(triangulated_point, idx)
                    added_idx.append(book_keeper.P.shape[1] - 1)  # add index of latest added point to visualize them
        return added_idx

    def wash_2(self, book_keeper, points_2d, points_3d, consensus_2d, view_id, thresh_vis_metric=0.14):
        """
        For each of the 2d-3d correspondences that don't appear in the consensus set:
        - if the 3d point is only visible in few amount of views: remove 3d point
        - if the 3d point has been visible for some time: remove the 2d observation of this point
        -----
        points_2d : 2xN
        points_3d : 3xN
        consensus_2d : 2xM
        consensus_3d : 3xM, M < N
        """
        removed_3d_idx = []
        for i in range(points_2d.shape[1]):
            if points_2d[:, i] not in consensus_2d.T:
                outlier_3d = points_3d[:, i].reshape(3, 1)
                vis_metric = book_keeper.get_visibility_metric(outlier_3d)
                if vis_metric < thresh_vis_metric:
                    # If the feature has only been visible in a few views, remove it completely
                    book_keeper.remove_point(outlier_3d)
                    removed_3d_idx.append(i)
                else:
                    # If it has been visible for some time, only the 2D observation should be removed
                    book_keeper.vis_matrix[view_id, i] = 0
        print(f'WASH 2: Removed outlier 3d points: {len(removed_3d_idx)}')
        return removed_3d_idx
