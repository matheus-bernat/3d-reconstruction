#!/usr/bin/env python3

import numpy as np
import random
from external import lab3


def is_rotation_matrix(R):
    """Returns if R is a rotation matrix."""
    prod_1 = np.round(R @ R.T, 3)
    prod_2 = np.round(R.T @ R, 3)
    det = np.round(np.linalg.det(R), 3)
    return (prod_1 == np.eye(3)).all() and (prod_2 == np.eye(3)).all() and (det == 1)


def c_norm(y, K):
    """C-normalize a """
    return np.linalg.inv(K) @ lab3.homog(y)


def d_norm(l):
    """D-normalize the line in dual homogeneous coordinates according to IREG 3.18"""
    assert(np.size(l) == 3)
    return 1/np.sqrt(np.square(l[0]) + np.square(l[1])) * l


def p_norm(v):
    if v[-1] == 0:
        raise ValueError('P-normalization failed due to division by 0')
    else:
        return v / v[-1]


def is_consistent_with_F(prev_point, new_point, F, epi_constraint_thresh=10):
    """
    Returns if (y1' * F * y2) < thresh. Same as doing (y2' * F' * y1) < thresh.
    """
    # Assure points are 2x1 vectors
    prev_point = prev_point.reshape(2, 1)
    new_point = new_point.reshape(2, 1)
    # Get dual homogeneous coordinates of line spanned by F * [y2; 1], and d-normalize it according p. 35 IREG.
    l = F @ lab3.homog(new_point)
    d_norm_l = d_norm(l)
    # Get the result of y1' * l. Ideally it should be 0, which means that the line passes through y1.
    res = lab3.homog(prev_point).T @ d_norm_l
    if res > epi_constraint_thresh:
        print(f'WARN: epipolar constraint not fulfilled by {res[0][0]:.1f}')
    else:
        # print(f'NICE: epipolar constraint fulfilled {res[0][0]:.1f}')
        pass
    return np.abs(res) < epi_constraint_thresh


def RANSAC_eight_point(iteration, threshold, point_set_l, point_set_r):
    max_inliner_num = 7
    sigma = 1e5
    for i in range(iteration):
        # Sample 8 points
        eight_samp_idc = np.array([random.sample(range(len(point_set_l)), 8)]).T
        pl = point_set_l[eight_samp_idc[:, 0], :].T
        pr = point_set_r [eight_samp_idc[:, 0], :].T
        F = lab3.fmatrix_stls(pl, pr)
        residuals = lab3.fmatrix_residuals(F, point_set_l.T, point_set_r.T)
        tmp_sum = np.sum(np.abs(residuals), axis=0)
        inlier_idc = np.argwhere(tmp_sum < threshold)[:, 0]
        inlier_num = np.shape(inlier_idc)[0]
        sigma_tmp = (np.sum((residuals ** 2)) / len(point_set_l)) ** 0.5

        if (inlier_num > max_inliner_num) or (inlier_num == max_inliner_num and sigma_tmp < sigma):
            max_inliner_num = inlier_num
            F_true = F
            sigma = sigma_tmp
            inliner_l = point_set_l[inlier_idc]
            inliner_r = point_set_r[inlier_idc]
            true_eight_points_l = pl
            true_eight_points_r = pr
    return F_true

def ssvd(M):
    """Toolbox page 109"""
    U, S, Vt = np.linalg.svd(M)
    V = Vt.T
    # U prime
    U0 = U[:, 0:-1]
    un = U[:, -1, None]
    U_prime = np.hstack([U0, np.linalg.det(U) * un])
    # V prime
    V0 = V[:, 0:-1]
    vm = V[:, -1, None]
    V_prime = np.hstack([V0, np.linalg.det(V) * vm])
    # Sigma prime
    sigma_r_prime = np.linalg.det(U) * np.linalg.det(V) * S[-1]
    S_prime = np.copy(S)
    S_prime[-1] = sigma_r_prime

    return U_prime, S_prime, V_prime


def print_epi_constraint_values_for_corresp(view_1, view_2, F):
    """Prints evaluations of the epipolar constraint y1' @ F @ y2"""
    for p1, p2 in zip(view_1.projections.T, view_2.projections.T):
        if p1[0] != -1 and p2[0] != -1:
            p1 = lab3.homog(p1).reshape(3, 1)
            p2 = lab3.homog(p2).reshape(3, 1)
            print(f'TEST: p1.T @ F @ p2 = {np.ndarray.item(p1.T @ d_norm(F @ p2)):.2f}')
