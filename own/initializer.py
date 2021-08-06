#!/usr/bin/env python3

import scipy.io
import numpy as np
import cv2
import random

from view import View
from external import lab3
from scipy.spatial.transform import Rotation as rot
from numpy.linalg import inv, svd, norm
from help_functions import ssvd, is_rotation_matrix, RANSAC_eight_point

class Initializer():
    """
    Class that initializes stuff...
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path : string
            Relative path to the .mat file containing newPs, newPoints3D matrices.
        gt_points_3d : [int, int, int]
            Matrix of shape [3, 676] containing the real locations of the interest points in 3D world coordinates.
        gt_views : View
            Ground truth View objects. Contains the view's real camera matrix.
        views : string
            View objects whose camera matrices will be estimated.
        """
        self.data = scipy.io.loadmat(path)
        self.gt_points_3d = np.array(self.data['newPoints3D'])
        self.gt_views, self.views = self.create_views()
        # Control that self.views isn't ground truth
        for view in self.views:
            assert(not view.is_gt)

    def create_views(self):
        """Creates ground truth and estimated View objects."""
        _gt_camera_matrices = self.data['newPs']
        nr_views = np.size(_gt_camera_matrices)
        gt_camera_matrices = np.array([_gt_camera_matrices[0][i] for i in range(nr_views)]) # 36 x [3x4] for Dino

        _projections = self.data['newPoints2D']
        projections_data = np.array([_projections[0][i] for i in range(nr_views)]) # 36 x [2x676] for Dino

        has_calculated_K = False
        K = None
        views = []
        gt_views = []
        for idx in range(nr_views):
            projections = projections_data[idx]
            image = cv2.cvtColor(cv2.imread('../external/dino/ppm/viff.0' + str(idx) + '.ppm'), cv2.COLOR_BGR2RGB)
            # Create GROUND TRUTH View object
            gt_camera_matrix = gt_camera_matrices[idx]
            new_gt_view = View(projections, id=idx, image=image, is_gt=True, camera_matrix=gt_camera_matrix)
            new_gt_view.camera_resection(gt_camera_matrix)
            gt_views.append(new_gt_view)

            # Calculate the intrinsics using one of the ground truth camera matrices.
            if not has_calculated_K:
                K = np.copy(new_gt_view.camera_intrinsics)
                has_calculated_K = True

            # Create View object which will be used for 3D reconstruction

            # Put skew to 0 for all of our camera intrinsics
            # TODO: REMOVE THIS IF WE DECIDE THAT SETTING SKEW TO 0 IS BAD
            K[0,1] = 0
            K[1,0] = 0

            new_view = View(projections, id=idx, image=image, K=K, is_gt=False)
            views.append(new_view)
        return gt_views, views

    def INIT1(self, all_views):
        """
        Chooses 2 initial views. Right now, simply takes nearby cameras (1st and 3rd) to assure they'll
        have a high number of correspondences and a larger distance (better for triangulation)
        Returns a View object.
        """
        idx_view_a = 0  # This camera will be used to calculate the relative camera matrix [R|t] for other 35 cameras.
        idx_view_b = 2
        return all_views[idx_view_a], all_views[idx_view_b]

    def INIT2(self, I_a, I_b):

        K = np.copy(I_a.camera_intrinsics)
        #K[0, 1] = 0 # remove skew

        proj_a = I_a.projections[:, :].T
        proj_b = I_b.projections[:, :].T

        corr_a = proj_a[np.logical_and(proj_a > 0, proj_b > 0)].reshape(-1, 2)
        corr_b = proj_b[np.logical_and(proj_a > 0, proj_b > 0)].reshape(-1, 2)

        print('INIT2: Number of corresp: ', corr_a.shape[0])

        # INIT2 - option 1, 2; opencv function
        #       - option 3; 8 point algorithm
        option = 4
        if(option == 1): # without skew
            K[0, 1] = 0  # remove skew
            cnormalized_a = inv(K) @ lab3.homog(corr_a.T)
            cnormalized_b = inv(K) @ lab3.homog(corr_b.T)

            E, mask = cv2.findEssentialMat(cnormalized_b[0:2, :].T, cnormalized_a[0:2, :].T, np.identity(3), method=cv2.RANSAC, prob=0.99, threshold=2)
        elif(option == 2): # with skew
            cnormalized_a = inv(K) @ lab3.homog(corr_a.T)
            cnormalized_b = inv(K) @ lab3.homog(corr_b.T)

            E, mask = cv2.findEssentialMat(cnormalized_b[0:2, :].T, cnormalized_a[0:2, :].T, np.identity(3),
                                           method=cv2.RANSAC, prob=0.99, threshold=2)
        elif(option == 3): # without skew
            # INIT2 - option 2; 8 point algorithm
            F = RANSAC_eight_point(iteration = 1000, threshold = 1, point_set_l = corr_a, point_set_r = corr_b)

            K = I_a.camera_intrinsics
            E = K.T @ F @ K

        elif(option == 4):
            E, mask = cv2.findEssentialMat(corr_b, corr_a, K, cv2.RANSAC, threshold=1)

        U, S, V = ssvd(E)

        assert(is_rotation_matrix(U))
        assert(is_rotation_matrix(V))

        """
        t = V[:, -1].reshape(3, 1)

        W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        C1 = np.hstack([np.eye(3), np.zeros([3, 1])])
        C2 = np.zeros([4, 3, 4])

        C2[0, :, :] = np.hstack([V @ W @ U.T, t])
        C2[1, :, :] = np.hstack([V @ W @ U.T, -t])
        C2[2, :, :] = np.hstack([V @ W.T @ U.T, t])
        C2[3, :, :] = np.hstack([V @ W.T @ U.T, -t])
        y1 = corr_a[0, :]
        y2 = corr_b[0, :]

        for i in range(4):
            p_3d_relative_first_camera = lab3.triangulate_optimal(C1, C2[i], y1, y2).reshape(3, 1)
            R = C2[i][:, 0:3]
            t = C2[i][:, 3].reshape(3, 1)
            p_3d_relative_second_camera = R @ p_3d_relative_first_camera + t
            if p_3d_relative_first_camera[2] > 0 and p_3d_relative_second_camera[2] > 0:
                print(p_3d_relative_first_camera[2] > 0)
                print(p_3d_relative_second_camera[2] > 0)
            return R, t, E
        """



        t_b = V[None, :, 2].T

        W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        R_b_1 = V @ W.T @ U.T
        R_b_2 = V @ W @ U.T

        # IREG algorithm 10.3: Determine a relative camera pose (R, t_hat) that is consistent with E

        R_a = np.identity(3)
        t_a = (np.zeros((t_b.shape)).T).reshape(-1, 1)
        C_prime_a = np.hstack((R_a, t_a))

        configs = np.zeros((4, 3, 4))
        configs[0] = np.hstack((R_b_1, t_b))
        configs[1] = np.hstack((R_b_1, -t_b))
        configs[2] = np.hstack((R_b_2, t_b))
        configs[3] = np.hstack((R_b_2, -t_b))

        num_inlier_opt = 0
        max_opt = -2
        x_bar = np.zeros((4, len(corr_a), 3))
        x_bar_prime = np.zeros((4, len(corr_a), 3))
        for i in range(4):
            C_a = K @ C_prime_a
            C_b = K @ configs[i]
            rigid_trans = np.vstack((configs[i], np.array([0, 0, 0, 1])))
            for j in range(len(corr_a)):
                x_bar[i, j] = lab3.triangulate_optimal(C_a, C_b, corr_a[j], corr_b[j])
                x_bar_prime[i, j] = (rigid_trans @ lab3.homog(x_bar[i, j]))[:3]
            p = x_bar[i, :, 2, None]
            q = x_bar_prime[i, :, 2, None]
            tmp = np.where(np.logical_and(p > 0, q > 0))
            if len(tmp[0]) > num_inlier_opt:
                max_opt = i
                num_inlier_opt = len(tmp[0])
        '''
        rand_idx = random.sample(range(len(corr_a)), 1)[0]
        for i in range(4):
            C_a = K @ C_prime_a
            C_b = K @ configs[i]
            rigid_trans = np.vstack((configs[i], np.array([0, 0, 0, 1])))
            #rand_idx = random.sample(range(len(corr_a)), 1)[0]
            x_bar = lab3.triangulate_optimal(C_a, C_b, corr_a[rand_idx], corr_b[rand_idx])
            x_bar_prime = (rigid_trans @ lab3.homog(x_bar))[:3]

            if (x_bar[2] > 0) and (x_bar_prime[2] > 0):
                max_opt = i
                break
        '''
        R = configs[max_opt][:, 0:3]
        t = configs[max_opt][:, -1].reshape(3, 1)

        return R, t, E
