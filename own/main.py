#!/usr/bin/env python3

import numpy as np
import time

from assembler import Assembler
from bookkeeper import Bookkeeper
from bundleadjustment import bundle_adjustment, wash1
from initializer import Initializer
from visualizer import Visualizer
from evaluator import Evaluator
from external import lab3
from help_functions import is_rotation_matrix
# from meshing import generate_mesh


def main():

    start = time.time()

    # Meshing
    _generate_mesh = False

    # Visualization
    _visualize_steps = True
    _color_bool = True  # True for estimating point color. False for blue points if performance is an issue.

    # HYPER-PARAMETERS
    _BA_learning_rate = 1e-4
    _BA_max_epochs = 100
    _BA_loss_thresh = 1e-4
    _BA_distance_thresh = 0.5
    _WASH1_thresh = 2
    _pnp_ransac_max_iter = 1000
    _pnp_inlier_thresh = 3            # max allowed reprojection error for a 3d points. Unit: pixels.
    _wash_2_thresh = 0.14             # between 0 and 1.
    _epi_constraint_thresh = 1

    text_file = open("images/parameters.txt", "w")
    text_file.write(f"_BA_learning_rate: {_BA_learning_rate}\n_BA_max_epochs: {_BA_max_epochs}\n"
                    f"_BA_loss_thresh: {_BA_loss_thresh}\n_WASH1_thresh: {_WASH1_thresh}")
    text_file.close()

    # Initialize objects
    init = Initializer(path='../external/dino/BAdino2.mat')
    book_keeper = Bookkeeper(all_views=init.views)
    ass = Assembler(K=init.views[0].camera_intrinsics)
    vis = Visualizer()

    # ------- INIT1 -------- Choose two initial cameras.
    view_1, view_2 = init.INIT1(init.views)
    book_keeper.add_view(view_1)
    book_keeper.add_view(view_2)
    view_1.set_camera_matrix(R=np.identity(3), t=np.zeros((3, 1)))

    # ------- INIT2 -------- Compute essential matrix and camera matrix for the second view
    R, t, E = init.INIT2(view_1, view_2)
    assert(is_rotation_matrix(R))
    view_2.set_camera_matrix(R, t)

    # ------- INIT3 -------- Triangulate common points
    F = ass.get_F_from_E(E)

    _ = ass.add_3d_points(view_1, view_2, book_keeper, F, _epi_constraint_thresh)

    used_ids = [view_1.id, view_2.id]

    print('\n=================================')

    cameras_left = True

    while cameras_left:

        print(f'USED VIEWS: {used_ids}')
        print(f'INFO: Number of points in P: {book_keeper.P.shape[1]}')

        # ------- BA & WASH1 --------
        bundle_adjustment(book_keeper, _BA_learning_rate, _BA_max_epochs, _BA_loss_thresh, _BA_distance_thresh)
        wash1(book_keeper, _WASH1_thresh)

        # ------- EXT1 -------- Choose new camera view
        prev_view, new_view = ass.choose_new_view(init.views)
        used_ids.append(new_view.id)

        # ------- EXT2 -------- Form set of 2d-3d correspondences
        points_2d, points_3d = ass.get_2d_3d_corresp(new_view, book_keeper.index_3d_2d, book_keeper.P)
        print('EXT2: Number of corresp: ', points_2d.shape[1])

        # ------- EXT3 -------- Find camera matrix from 2d-3d correspondences & consensus set
        R, t, consensus_2d, consensus_3d = ass.robust_pnp(points_2d, points_3d, _pnp_ransac_max_iter, _pnp_inlier_thresh)

        # ------- EXT4 -------- Add new view to Q and add new view's projections to book_matrix
        new_view.set_camera_matrix(R, t)
        book_keeper.add_view(new_view)

        # ------- EXT5 -------- Determine E between current 2 views, triangulate points and add them to P
        F = lab3.fmatrix_from_cameras(prev_view.camera_matrix, new_view.camera_matrix)
        F = F / F[-1, -1]  # different to divide by last element?
        added_3d_idx = ass.add_3d_points(prev_view, new_view, book_keeper, F, _epi_constraint_thresh)

        # ------- WASH2 -------- Clean P from bad 3d triangulations
        removed_3d_idx = ass.wash_2(book_keeper, points_2d, points_3d, consensus_2d, new_view.id, _wash_2_thresh)
        book_keeper.age_points()

        cameras_left = len(book_keeper.Q) < len(init.views)

        total_3d_points_added = len(added_3d_idx) - len(removed_3d_idx)

        print(f'INFO: Total 3d points added: {total_3d_points_added}')

        if _visualize_steps:
            added_3d_points = book_keeper.P[:, -total_3d_points_added:]
            # vis.visualize_reprojection(new_view, added_3d_points)
            vis.visualize_3d_points(book_keeper, total_3d_points_added, estimate_color=_color_bool)

        print('\n=================================')

    # last bundle adjustment and wash1
    bundle_adjustment(book_keeper, _BA_learning_rate, _BA_max_epochs, _BA_loss_thresh, _BA_distance_thresh)
    wash1(book_keeper, _WASH1_thresh)

    end = time.time()
    print(f'Total elapsed time: {end - start}')
    print(f'Total triangulated points: {book_keeper.P.shape[1]}')

    # Construct and show 3d reconstruction
    print("\nWOW, LOOK AT THE DINO!")

    if _generate_mesh:
        generate_mesh(book_keeper)

    vis.visualize_3d_points(book_keeper, total_3d_points_added, estimate_color=_color_bool)

    # ------- EVALUATION -------- Save the 3d point clouds for evaluation
    est_rotations = np.zeros([36, 9])
    est_translations = np.zeros([36, 3])
    gt_rotations = np.zeros([36, 9])
    gt_translations = np.zeros([36, 3])
    for i in range(len(book_keeper.Q)):
        est_rotations[book_keeper.Q[i].id, ...] = book_keeper.Q[i].rotation_matrix.reshape(9, )
        est_translations[book_keeper.Q[i].id, ...] = book_keeper.Q[i].translation_vector.reshape(3, )
        gt_rotations[i, ...] = init.gt_views[i].rotation_matrix.reshape(9, )
        gt_translations[i, ...] = init.gt_views[i].translation_vector.reshape(3, )
    np.savetxt(fname='output/triang_points.out', X=book_keeper.P)
    np.savetxt(fname='output/gt_points.out', X=init.gt_points_3d)
    np.savetxt(fname='output/est_rotations.out', X=est_rotations)
    np.savetxt(fname='output/est_translations.out', X=est_translations)
    np.savetxt(fname='output/gt_rotations.out', X=gt_rotations)
    np.savetxt(fname='output/gt_translations.out', X=gt_translations)


def evaluation():
    eval = Evaluator()
    # gt_points = np.loadtxt(fname='output/gt_points.out').T  # 3x676
    # triang_points = np.loadtxt(fname='output/triang_points.out')
    est_rotations = np.loadtxt(fname='output/est_rotations.out').reshape(36, 3, 3)
    est_translations = np.loadtxt(fname='output/est_translations.out').reshape(36, 3, 1)
    gt_rotations = np.loadtxt(fname='output/gt_rotations.out').reshape(36, 3, 3)
    gt_translations = np.loadtxt(fname='output/gt_translations.out').reshape(36, 3, 1)
    eval.estimate_rotation(gt_translations, est_translations)
    eval.estimate_scale()
    eval.estimate_translation()
    eval.map_camera_center(est_rotations, est_translations)
    eval.visualize_camera_centers(gt_rotations, gt_translations)
    eval.compute_errors(gt_rotations, gt_translations)

if __name__ == '__main__':
    main()
    # evaluation()
