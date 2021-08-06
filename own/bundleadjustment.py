#!/usr/bin/env python3
import torch
from scipy.spatial.transform import Rotation
import numpy as np
import sys
sys.path.append('../')
from external.lab3 import project
import external.pytorch3d_helpers as py3d
import warnings


def mat2aa(R):  # takes R as matrix and returns as axis angle
    return Rotation.from_matrix(R).as_rotvec()


def aa2mat(R):  # reverse from mat2aa
    return Rotation.from_rotvec(R).as_matrix()


def project_tensor(x, C):

    # to homogeneous
    X = torch.nn.functional.pad(input=x, pad=(0, 0, 0, 1), mode='constant', value=1)
    y = C @ X

    # normalize
    y = y / y[2]
    return y[:2]


# model for the optimizer
class Model(torch.nn.Module):
    def __init__(self, Q, P, P_index, K, BK_mat, vis_mat):
        super(Model, self).__init__()
        self.Q = Q
        self.P = P
        self.P_index = P_index
        self.K = K
        self.BK_mat = BK_mat
        self.vis_mat = vis_mat

        # check if GPU available
        #if torch.cuda.is_available():
        #dev = "cuda"
        #else:
        dev = "cpu"
            # print('Cuda not available, will run bundle adjustment on CPU')
        device = torch.device(dev)

        # get all rotations in axis angle and other parameters in tensor form
        rot_vecs = np.empty((3,0))
        t_vecs = np.empty((3,0))
        for i in range(1,len(self.Q)): # do not optimize reference cam
            view = self.Q[i]
            rot_vecs = np.hstack((rot_vecs, np.atleast_2d(mat2aa(view.rotation_matrix)).T)) # (3, N) N is camera views
            t_vecs = np.hstack((t_vecs, view.translation_vector)) # (3, N)
        self.R_param = torch.nn.Parameter(torch.from_numpy(rot_vecs))
        self.R_param.requires_grad = True
        self.R_param = self.R_param.to(device)
        self.T_param = torch.nn.Parameter(torch.from_numpy(t_vecs))
        self.T_param.requires_grad = True
        self.T_param = self.T_param.to(device)
        self.P_param = torch.nn.Parameter(torch.from_numpy(self.P))
        self.P_param.requires_grad = True
        self.P_param = self.P_param.to(device)

    def error_function(self):
        error = 0
        # form the K matrix
        K_torch = torch.from_numpy(self.K)

        for i in range(len(self.Q)):
            bk_id = self.Q[i].id
            # form the projection matrix, do not optimize the reference cam
            if i > 0:
                C = K_torch @ torch.hstack((py3d.axis_angle_to_matrix(self.R_param[:,i-1]),torch.atleast_2d(self.T_param[:,i-1]).T))
            else:
                C = torch.from_numpy(self.Q[0].camera_matrix)
            # Rearrange view correspondences to match P_index and filter out non-visible points
            # read row for view
            q = self.BK_mat[bk_id, :]
            v = self.vis_mat[bk_id, :]
            # rearrange according to P_index
            q = q[self.P_index].T
            v = v[self.P_index]
            # only get visible ones
            p = self.P_param[:, v == 1]

            y = torch.from_numpy(q[:, v == 1])

            y_proj = project_tensor(p, C)

            res_mse = torch.mean(torch.sum(torch.square(y-y_proj), dim=0))
            # print(f'reprojection MSE for view {i}:', res_mse.item())
            error += res_mse
        # print('Sum of MSE:', error.item())
        return error


def bundle_adjustment(Book_keeper, learning_rate=1e-3, max_epochs=1000, loss_diff_thresh = 1e-4, distance_thresh = 0.5):
    # read from book_keeper
    Q = Book_keeper.Q
    K = Q[0].camera_intrinsics
    P = Book_keeper.P
    old_P = np.copy(P)
    P_index = Book_keeper.index_3d_2d
    BK_mat = Book_keeper.book_matrix
    vis_mat = Book_keeper.vis_matrix

    # init model and optimizer
    model = Model(Q, P, P_index, K, BK_mat, vis_mat)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # optimize
    prev_loss = np.inf
    for _ in range(max_epochs):
        loss = model.error_function()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if np.abs(prev_loss-loss.item()) < loss_diff_thresh:
            break
        else:
            prev_loss = loss.item()

    print(f'BA: Sum of squared reproj error: {loss.item():.4f}')

    R_optim = model.R_param.detach().cpu().numpy()
    t_optim = model.T_param.detach().cpu().numpy()

    # update Q for i>0
    for i in range(1, len(Q)):
        Q[i].set_camera_matrix(aa2mat(R_optim[:, i-1]),np.atleast_2d(t_optim[:, i-1]).T)

    # update P
    optimal_P = model.P_param.detach().cpu().numpy()
    Book_keeper.P = optimal_P
    # wash points that moved too far
    dist_moved = np.sqrt(np.sum(np.square(optimal_P - old_P), axis=0))
    # print('distance moved: ', dist_moved)
    outliers = np.where(np.greater(dist_moved, distance_thresh))[0]

    delete_ps = Book_keeper.P[:, outliers]
    for i in range(len(outliers)):
        delete_p = delete_ps[:, i]
        Book_keeper.remove_point(delete_p)

    if len(outliers) > 0:
        print('Points washed due to far movement in BA: ', outliers)

def wash1(Book_keeper, threshold=np.inf):
    """
    Washes any 3D points that have too large of a projection error
    """
    Q = Book_keeper.Q
    P = Book_keeper.P
    P_index = Book_keeper.index_3d_2d
    BK_mat = Book_keeper.book_matrix
    vis_mat = Book_keeper.vis_matrix

    # initialize error array for each 3D point
    reprojection_errors = np.zeros((len(Q),P.shape[1]))

    # add up squared sum reprojection errors
    for i in range(len(Q)):
        cur_view = Q[i]
        bk_id = cur_view.id

        # Rearrange view correspondences to match P_index and filter out non-visible points
        # read row for view
        q = BK_mat[bk_id, :]
        v = vis_mat[bk_id, :]

        # rearrange according to P_index
        q = q[P_index].T
        v = v[P_index]

        # only get visible ones
        p_ind = np.array(P_index)[v == 1]
        p = P[:, v == 1]
        y_proj = project(p, cur_view.camera_matrix)
        y = q[:, v == 1]

        residual = y - y_proj
        residual_sq_sum = np.sum(np.square(residual), axis=0)
        for j in range(len(p_ind)):
            reprojection_errors[i] += np.where(P_index == p_ind[j], residual_sq_sum[j], 0)
    reprojection_errors[reprojection_errors == 0] = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # ignore a warning with nanmean
        reprojection_errors_mean = np.nanmean(reprojection_errors, axis=0)

    # get indices and delete 3D points
    outliers = np.where(np.greater(reprojection_errors_mean,threshold))[0]
    delete_ps = Book_keeper.P[:, outliers]
    for i in range(len(outliers)):
        delete_p = delete_ps[:, i]
        Book_keeper.remove_point(delete_p)

    if len(outliers) > 0:
        print('Points washed in WASH1: ', outliers)