#!/usr/bin/env python3

import numpy as np
from visualizer import Visualizer
from help_functions import ssvd


class Evaluator():
    """
    Declares functions used for evaluation.
    """
    def __init__(self):
        self.gt_centroid = None
        self.est_centroid = None
        self.gt_bary = None
        self.est_bary = None
        self.R = None
        self.t = None
        self.s = None
        self.R_mapped = None
        self.t_mapped = None
        self.MAE_t_err = None
        self.MAE_R_err = None
        self.vis = Visualizer()

    def estimate_rotation(self, gt_trans, est_trans):
        """Estimates rotation R according to IREG algorithm 15.4"""
        self.gt_centroid = 1 / gt_trans.shape[0] * np.sum(gt_trans, axis=0)
        self.est_centroid = 1 / est_trans.shape[0] * np.sum(est_trans, axis=0)

        self.gt_bary = (gt_trans - self.gt_centroid).reshape(36,3).T
        self.est_bary = (est_trans - self.est_centroid).reshape(36,3).T

        U, _, V = np.linalg.svd(self.est_bary @ self.gt_bary.T)
        self.R = V @ U.T

    def estimate_scale(self):
        """
        Estimate scale according to Horn section 3D
        Ref: http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf
        """
        num = 0
        for i in range(self.gt_bary.shape[1]):
            sth = np.dot(self.gt_bary[:, i], self.R @ self.est_bary[:, i])
            num += sth
        den = np.sum(np.linalg.norm(self.est_bary, axis=0)**2)
        self.s = num / den

    def estimate_translation(self):
        """
        Estimate translation according to Horn 3B
        Ref: http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf
        """
        self.t = self.gt_centroid - self.s * self.R @ self.est_centroid

    def map_camera_center(self, R_est, t_est):
        """
        Maps the estimated camera centers using the calculated scale s, the global translation R
        and global translation t.
        ------
        Inputs:
            R_est : float 36x3x3
            t_est : float 36x3x1
        """
        t_mapped = np.zeros([36, 3, 1])
        R_mapped = np.zeros([36, 3, 3])
        for i in range(R_est.shape[0]):
            t_mapped[i, ...] = self.s * self.R @ t_est[i, ...] + self.t
            R_mapped[i, ...] = R_est[i, ...] @ self.R.T
        self.t_mapped = t_mapped
        self.R_mapped = R_mapped

    def visualize_camera_centers(self, R_gt, t_gt):
        self.vis.visualize_camera_centers(self.R_mapped, self.t_mapped, R_gt, t_gt)

    def compute_errors(self, R_gt, t_gt):
        """
        estimates error between ground truth of translations and rotations and mapped translations and rotations.
        ------
        Inputs:
            R_gt : float 36x3x3
            t_gt : float 36x3x1
        """
        t_err = np.zeros([36, 1])
        R_err = np.zeros([36, 1])
        for i in range(len(t_err)):
            t_err[i] = np.linalg.norm(t_gt - self.t_mapped)
            R_err[i] = 2 * np.arcsin(np.linalg.norm(R_gt[i, ...] - self.R_mapped[i, ...]) / np.sqrt(8))
        self.MAE_t_err = np.sum(np.abs(t_err)) / len(t_err)
        print(f'Translation mean absolute error: {self.MAE_t_err}')
        self.MAE_R_err = np.sum(np.abs(R_err)) / len(R_err)
        print(f'Rotation mean absolute error: {self.MAE_R_err}')