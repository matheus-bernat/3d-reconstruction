#!/usr/bin/env python3

import numpy as np
from help_functions import is_rotation_matrix
from external import lab3


class View():
    """
    Class that represents a view (i.e. a camera). It contains the camera's intrinsic
    properties and projections.
    """
    def __init__(self, projections, id, image=None, K=None, is_gt=False, camera_matrix=None):
        """
        Members
        ----------
        camera_intrinsics : [int, int]
            [3x3] matrix representing the camera matrix instrinsics
        camera_position : [int, int]
            [3x4] matrix representing the camera position
        camera_matrix :
            [3x4] camera projection matrix for the view
        image : [int, int]
            Image taken by this view. Used for debugging purposes
        projections : [int, int] 2 x 676
            2D projections of the interest points given by the dataset
        """
        self.is_gt = is_gt
        self.image = image
        self.id = id
        self.camera_intrinsics = K                 # K. Same for all cameras. Given.
        self.projections = projections             # 2x676
        self.rotation_matrix = None                # [R | t]
        self.translation_vector = None
        self.camera_matrix = camera_matrix         # C = K @ [R|t]
        self.used = False
        self.camera_center = None

    def camera_resection(self, C):
        """
        Camera resectioning according to Algorithm 8.1 in IREG. Sets self.camera_intrinsics
        and self.camera_position.
        """
        A = C[:, 0:3]
        b = C[:, 3]
        U, Q = self.qr_factorize(A)
        t = np.linalg.inv(U) @ b
        t = np.expand_dims(t, axis=1)
        U = np.divide(U, U[2, 2])
        D = np.diag(np.sign(np.diag(U)))
        K = U @ D
        if np.linalg.det(D) == 1:
            R = D @ Q
            t = D @ t
        else:
            R = -D @ Q
            t = -D @ t

        assert(np.allclose(K, np.triu(K)))        # assure that K is upper triangular
        assert(is_rotation_matrix(R))        # assure that R is SO(3)

        self.camera_intrinsics = K                # 3x3
        self.rotation_matrix = R                  # 3x3
        self.translation_vector = t               # 3x1
        self.camera_matrix = K @ np.hstack((R, t))

        # Alternative (probably safer?)
        # K, R, t, _, _, _, _ = cv.decomposeProjectionMatrix(C)

    def set_camera_matrix(self, R, t):
        assert (is_rotation_matrix(R))
        self.rotation_matrix = R
        self.translation_vector = t
        self.camera_matrix = self.camera_intrinsics @ np.hstack((R, t))
        self.camera_center = self.get_camera_center()

    def get_camera_center(self):
        """
        Get the camera center of this view from the camera matrix. The camera center is
        a right null vector of the camera matrix (C @ n = 0), so the last column of V, which
        corresponds to the least singular value (0), is a null vector of C.
        Ref: IREG page 96.
        """
        # U, S, V_t = np.linalg.svd(self.camera_matrix)
        # camera_center = V_t[-1, :].reshape(4, 1)  # 4x1
        # camera_center = p_norm(camera_center)  # let the last coordinate be 1 so that it's in cartesian coordinates

        camera_center = -self.rotation_matrix.T @ self.translation_vector
        assert (np.round(self.camera_matrix @ lab3.homog(camera_center), 5).all() == 0)

        return camera_center

    def qr_factorize(self, M):
        """
        Special QR factorization of matrix M [3x3] according to Algorithm 3 in 
        Mathematical Toolbox page 122. 
        http://www.cvl.isy.liu.se/research/publications/PRE/0.40/main-pre-0.40.pdf.
        """
        m2 = M[1, :]        # second last row in M
        m3 = M[2, :]        # last row in M
        q3 = np.divide(m3, np.linalg.norm(m3))
        q2 = (m2 - q3 * np.dot(q3, m2)) / np.sqrt(np.linalg.norm(m2)**2 - np.dot(q3, m2)**2)
        q1 = np.cross(q2, q3)
        Q = np.vstack((q1, q2, q3))
        U = M @ Q.T
        return U, Q
