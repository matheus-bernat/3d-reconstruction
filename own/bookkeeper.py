#!/usr/bin/env python3

import numpy as np

class Bookkeeper():
    """
    Class to bookkeep the data matrices as well as lists of views Q and 3d points P
    """
    def __init__(self, all_views):
        """
        Members
        ----------
        P : [int, int]
            3xN array of estimated 3D points
        Q : [View]
            list of used View objects
        book_matrix : len(Q)xN matrix
            matrix consisting of 2D coordinates of projections of 3D point p in view q, 0 means not visible
        vis_matrix : len(Q)xN matrix
            binary matrix, 1 if 3D point p is visible in view q
        index_3d_2d : [int]
            List of indexes telling to which column in book_matrix a certain 3d point correspond to. E.g. if
            index_3d_2d = [5, 24, 0], then the 1st point in P corresponds to the 2d points in the 5th column in
            book_matrix, etc.
        """

        self.P = np.empty([3,0])
        self.index_3d_2d = np.empty([1, 0])
        self.P_age = np.empty([1,0])
        self.Q = []

        self.book_matrix = np.zeros((len(all_views), all_views[0].projections.shape[1], 2)) # 36 x 676 x 2
        self.vis_matrix = np.zeros((len(all_views), all_views[0].projections.shape[1])) # 36 x 676
        for idx in range(len(all_views)):
            self.book_matrix[idx] = np.array(all_views[idx].projections).T
            self.vis_matrix[idx] = np.where(all_views[idx].projections[0] == -1, 0, 1).T

    def add_view(self, new_view):
        """
        Adds a view object to the book-keeping. Adds new_view to the list of already used views, Q,
        and also adds new_view's object
        """
        self.Q.append(new_view)
        new_view.used = True

    def add_point(self, point_3d, idx_3d_2d):
        """
        Adds 3d point to the book-keeping.
        ------
        Input:
            point_3d : 3x1 numpy array
        """
        point_3d = point_3d.reshape(3, 1)
        self.P = np.hstack((self.P, point_3d)) if self.P.size else point_3d
        self.P_age = np.hstack((self.P_age, 0)) if self.P_age.size else np.array([0])
        self.index_3d_2d = np.hstack([self.index_3d_2d, idx_3d_2d]) if self.index_3d_2d.size else np.array([idx_3d_2d])

    def age_points(self):
        self.P_age += 1

    def get_visibility_metric(self, point_3d):
        """
        The visibility metric is defined as the percentage of views that can see point_3d.
        As a probability, it must lie between 0 and 1. Used in WASH_2.
        """
        p_idx = np.where((self.P.T == point_3d.T).all(axis=1))
        p_idx = int(p_idx[0])
        vis_metric = 0
        for view in self.Q:
            vis_metric += self.vis_matrix[view.id, self.index_3d_2d[p_idx]]
        assert(vis_metric / len(self.Q) <= 1)
        return vis_metric / len(self.Q)

    def remove_point(self, point_3d):
        """
        Remove 3d point from list of triangulated points, P.
        """
        p_idx = np.where((self.P.T == point_3d.T).all(axis=1))
        p_idx = int(p_idx[0])
        self.vis_matrix[:, self.index_3d_2d[p_idx]] = 0  # set this 3d point as invisible for all views
        self.P = np.delete(self.P, p_idx, axis=1)
        self.P_age = np.delete(self.P_age, p_idx)
        self.index_3d_2d = np.delete(self.index_3d_2d, p_idx)
        assert(self.P.shape[1] == len(self.index_3d_2d))
        assert(self.P.shape[1] == self.P_age.shape[0])


