# encoding: utf8
from __future__ import print_function, division
"""
Utility functions for Computer Exercise 3: Optimization in TSBB15 Computer Vision

The functions in this module mostly map directly to their
equivalent in the MATLAB toolbox for CE3.


Written by Hannes Ovrén, 2016
Minor updates by Johan Edstedt, 2020
"""

import os
import warnings
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import ConnectionPatch
from scipy.ndimage.filters import maximum_filter
from PIL import Image

# from tsbb15_labs import IMAGE_DIRECTORY


# Handle both OpenCV 2.4 and 3+
try:
    IMREAD_COLOR = cv2.IMREAD_COLOR
except AttributeError:
    IMREAD_COLOR = cv2.CV_LOAD_IMAGE_COLOR


#LAB3_IMAGE_DIRECTORY = Path(os.path.join(IMAGE_DIRECTORY,'lab3'))


def load_image_grayscale(path):
    "Load a grayscale image by path"
    return np.asarray(Image.open(path).convert('L'))


def load_stereo_pair():
    """Load stereo image pair
    
    Returns
    ------------------
    img1: np.ndarray
        First image in pair
    img2: np.ndarray
        Second image in pair
    """
    pair = 'img'
    return [
        load_image_grayscale(LAB3_IMAGE_DIRECTORY / f'{pair}{i}.png') 
        for i in (1, 2)
    ]


def homog(x):
    """Homogenous representation of a N-D point
    
    Parameters
    ----------------
    x : (N, 1) or (N, ) array
        The N-dimensional point
    
    Returns
    ----------------
    xh : (N+1, 1) or (N+1, ) array
        The point x with an extra row with a '1' added
    """
    is2d = (x.ndim == 2)
    if not is2d:
        x = x.reshape(-1,1)
    d, n = x.shape
    X = np.empty((d+1, n))
    X[:-1, :] = x
    X[-1, :] = 1
    return X if is2d else X.ravel()


def project(x, C):
    """Project 3D point
    
    Parameters
    --------------
    x : (3,1) or (3,) array
        A 3D point in world coordinates
    C : (3, 4) matrix
        Camera projection matrix
    
    Returns
    -------------------
    y : (2, 1) array
        Projected image point
    """
    if not C.shape == (3,4):
        raise ValueError('C is not a valid camera matrix')
    X = homog(x)
    y = np.dot(C, X)
    y /= y[2]
    return y[:2]


def load_image(fpath):
    """Load an image from file path
    
    Parameters
    ----------------
    fpath : string
        Path to the image
    
    Returns
    ----------------
    im : (M, N, 3)
        Numpy ndarray representation of the image (BGR color)
    """
    im = cv2.imread(fpath, IMREAD_COLOR)
    if im is None:
        raise IOError("Failed to load {}".format(fpath))
    return im


def rgb2gray(img):
    """Convert color image (BGR) to grayscale
    
    Please note that this only handles BGR formatted images
    because this is what load_image() (and the OpenCV backend) uses.
    
    Parameters
    ---------------
    img : (M, N, 3) array
        The color image, in BGR format
    
    Returns
    ---------------
    (M, N) array
        Grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def cross_matrix(v):
    """Compute cross product matrix for a 3D vector
    
    Parameters
    --------------
    v : (3,) array
        The input vector
        
    returns
    --------------
    V_x : (3,3) array
        The cross product matrix of v such that V_x b == v x b
    """
    v = v.ravel()
    if not v.size == 3:
        raise ValueError('Can only handle 3D vectors')
    
    return np.array([[0, -v[2], v[1]],
                     [v[2],  0, -v[0]],
                     [-v[1], v[0], 0]])


def harris(image, block_size, kernel_size):
    """ Compute Harris response
    
    Parameters
    ---------------
    image : (M, N, 3) or (M,N) ndarray (dtype uint8)
         Input RGB or gray scale image
    block_size : int
        Side of square neighborhood for eigenvalue computation
    kernel_size : int
        Side of square gradient filter kernels. Must be 1, 3, 5 or 7
        
    Returns
    ------------------
    H : (M, N) ndarray (dtype float32)
        The Harris response image, with negative values set to zero
    """
    if kernel_size not in (1, 3, 5, 7):
        raise ValueError('kernel_size must be 1, 3, 5, or 7')
    if not image.dtype == np.uint8:
        raise ValueError('Image type must be 8-bit unsigned (np.uint8)')
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    harris_param = 0.04 # "Standard value"
    harris = cv2.cornerHarris(image, block_size, kernel_size, harris_param)
    harris[harris < 0] = 0
    return harris


def non_max_suppression(image, window_size):
    """ Suppress pixels which are not local maxima
    
    Parameters
    ----------------
    image : (M, N) ndarray
        Input image
    window_size : int
        Size of the window a pixel must be the maximum of to be kept. Must be odd.
        
    Returns
    ----------------
    image, with non-max pixels set to zero.
    """
    if not window_size % 2 == 1:
        raise ValueError('window_size must be odd')
    if not image.ndim == 2:
        raise ValueError('image must be single channel (i.e. a 2D array)')
        
    h, w = image.shape
    m = int((window_size - 1) / 2)
    mask = np.zeros_like(image)
    mask[m:h-m,m:w-m] = 1 # only keep valid max positions.
    max_filt_im = maximum_filter(image,size=window_size)*mask
    non_max_suppressed = (max_filt_im==image)*image
    return non_max_suppressed


def fmatrix_residuals(F, x, y):
    """Calculate residuals for fundamental matrix and image points
    
    Calculates the distances between the epipolar lines and the supplied points.
    
    Parameters
    -------------------
    F : (3,3) ndarray
        Fundamental matrix such that x.T * F * y = 0
    x : (2, N) ndarray
        Points in left image
    y : (2, N) ndarray
        Points in right image
    
    Returns
    -------------------
    residuals : (2, N) ndarray
        Signed distance from x (dim 0) and y (dim 1) to epipolar lines
    """
    if not x.shape == y.shape:
        raise ValueError('x and y must have same sizes')
    
    x = homog(x)
    y = homog(y)

    l1 = np.dot(F, y)
    l2 = np.dot(F.T, x)
    
    #normalize = lambda l: l / np.sqrt(l[0,:]**2 + l[1,:]**2)
    #l1 = normalize(l1)
    #l2 = normalize(l2) 
    l1s = np.sqrt(l1[0,:]**2 + l1[1,:]**2)
    l2s = np.sqrt(l2[0,:]**2 + l2[1,:]**2)
    
    #res1 = np.sum(l1 * x, axis=0)
    #res2 = np.sum(l2 * y, axis=0)
    res1 = np.sum(l1 * x, axis=0) / l1s
    res2 = np.sum(l2 * y, axis=0) / l2s
    
    return np.vstack((res1, res2))


def fmatrix_residuals_gs(params, pl, pr):
    """
    Parameters
    -------------------
    params : (12+3N, ) array
        Parameter vector of first camera and all 3D points.
        Given a (3,4) matrix C1 and (3, N) matrix X it is created
        as params = np.hstack((C1.ravel(), X.T.ravel())).
        This means that the camera matrix is stored row-first,
        but points column-first. Note the transpose of X!
    pl : (2, N) array
        Left image points
    pr : (2, N) array
        Right image points
    
    Returns
    ----------------
    residuals : (4N,) ndarray
        Order of residuals: leftx, lefty, rightx, righty
    """
    # Extract cameras
    C1 = params[:12].reshape(3,4)
    C2 = np.zeros((3,4)); C2[:3, :3] = np.eye(3)
    
    # Extract 3D points
    X = params[12:].reshape(-1, 3).T
    
    if not X.shape[1] == pl.shape[1]:
        raise ValueError('Wrong size of parameter vector')
    
    yl = project(X, C1)
    yr = project(X, C2)
    
    r1 = pl - yl
    r2 = pr - yr
    
    return np.concatenate((r1.ravel(), r2.ravel()))


def fmatrix_stls(pl, pr):
    """Estimate fundamental matrix using 8-point algorithm
    
    Parameters
    ---------------
    pl : (2, N) ndarray
        Left image coordinates
    pr : (2, N) ndarray
        Right image coordinates
        
    Returns
    --------------
    Fundamental matrix F
    """
    if not pl.shape == pr.shape:
        raise ValueError('pl and pr must have same shape')
    
    _, N = pl.shape
    
    def scaling_homography(x):
        xm = np.mean(x, axis=1)
        x_norm = x - xm.reshape(-1,1) # Remove mean
        L = np.sqrt(1. / 2. / N * np.sum(x_norm**2))
        H = np.array([[1. / L, 0        , -xm[0] / L],
                      [0.    , 1. / L   , -xm[1] / L],
                      [0.    , 0.       , 1.         ]])
        return H
    
    S = scaling_homography(pl)
    T = scaling_homography(pr)
    
    # Map points through S and T respectively
    def map_homography(p, H):    
        x = p[0]
        y = p[1]
        xh = x * H[0,0] + y * H[0,1] + H[0, 2]
        yh = x * H[1,0] + y * H[1,1] + H[1, 2]
        return xh, yh
    
    lhx, lhy = map_homography(pl, S)
    rhx, rhy = map_homography(pr, T)
    
    # Generate matrix A for which A * vec(F) = 0
    X, Y = lhx, lhy
    x, y = rhx, rhy
    A = np.vstack((X*x, X*y, X, Y*x, Y*y, Y, x, y, np.ones((1, N))))
    A = A.T
    
    U, s, V = np.linalg.svd(A); # Note: V already transposed
    Fs = V[-1, :].reshape(3,3)
    
    # Enforce rank 2
    U, s, V = np.linalg.svd(Fs) # Note: V already transposed
    D = np.diag(s)
    D[2,2] = 0
    Fs = np.dot(U, np.dot(D, V))
    
    # Undo scaling
    F = np.dot(S.T, np.dot(Fs, T))
    
    return F


def fmatrix_from_cameras(C1, C2):
    """Fundamental matrix from camera pair
    
    Parameters
    ------------------
    C1 : (3, 4) array
        Camera 1
    C2 : (3, 4) array
        Camera 2
        
    Returns
    ---------------------
    F : (3,3) array
        Fundamental matrix corresponding to C1 and C2
    """
    U, s, V = np.linalg.svd(C2) # Note: C2 = U S V  (V already transposed)
    n = V[3, :]
    e = np.dot(C1, n)
    C2pinv = np.linalg.pinv(C2)
    F = np.dot(cross_matrix(e), np.dot(C1, C2pinv))
    return F


def fmatrix_cameras(F):
    """Camera pairs for a fundamental matrix
    
    This returns one possible combination of cameras
    consistent with the fundamental matrix.
    The second camera is always fixed to be C2 = [I | 0].
    
    Parameters
    --------------
    F : (3,3) array
        Fundamental matrix
    
    Returns
    -------------
    C1 : (3, 4) array
        The first camera
    C2 : (3, 4) array
        The second camera. This is always [I | 0]
    """
    C2 = np.zeros((3,4), dtype='double')
    C2[:3,:3] = np.eye(3)
    
    U, s, V = np.linalg.svd(F)
    e1 = U[:, -1]
    A = np.dot(cross_matrix(e1), F)
    C1 = np.hstack((A, e1.reshape(-1,1)))
    
    return C1, C2


def triangulate_optimal(C1, C2, x1, x2):
    """Optimal trinagulation of 3D point
    
    Parameters
    ------------------
    C1 : (3, 4) array
        First camera
    C2 : (3, 4) array
        Second camera
    x1 : (2,) array
        Image coordinates in first camera
    x2 : (2,) array
        Image coordinates in second camera
    
    Returns
    ------------------
    X : (3, 1) array
        The triangulated 3D point
    """
    move_orig = lambda x: np.array([[1., 0., x[0]],[0., 1., x[1]], [0., 0., 1.]])
    T1 = move_orig(x1)
    T2 = move_orig(x2)
    
    # Find and transform F
    F = fmatrix_from_cameras(C1, C2)
    F = np.dot(T1.T, np.dot(F, T2))
    
    # Extract epipoles
    # Normalize to construct rotation matrix
    e1, e2 = fmatrix_epipoles(F)
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    
    R_from_epipole = lambda e: np.array([[e[0], e[1], 0],[-e[1], e[0], 0],[0,0,1]])
    R1 = R_from_epipole(e1)
    R2 = R_from_epipole(e2)
    
    F = np.dot(R1, np.dot(F, R2.T))
    
    # Build polynomial, with code from Klas Nordberg
    f1 = f2 = 1 # Note: Matlab implementation assumed e1, e2 homogeneous, and fk=e[-1]
    a = F[1,1]
    b = F[1,2]
    c = F[2,1]
    d = F[2,2]
    k1 = b * c - a * d
    g = [a * c * k1 * f2**4, #coefficient for t^6
         (a**2 + c**2 * f1**2)**2 + k1 * (b * c + a * d) * f2**4, #coefficient for t^5
         4 * (a**2 + c**2 * f1**2) * (a * b + c * d * f1**2) + \
         2 * a * c * k1 * f2**2 + b * d * k1 * f2**4,         
         2 * (4 * a * b * c * d * f1**2 + a**2 * (3 * b**2 + d**2 * (f1-f2) * (f1 + f2)) + \
         c**2 * (3 * d**2 * f1**4 + b**2 * (f1**2 + f2**2))),
         -a**2 * c * d + a * b *(4 * b**2 + c**2 + 4*d**2 * f1**2 - 2 * d**2 * f2**2) + \
         2 * c * d * (2 * d**2 * f1**4 + b**2 * (2 * f1**2 + f2**2)),
         b**4 - a**2 * d**2 + d**4 * f1**4 + b**2 * (c**2 + 2 * d**2 * f1**2),
         b * d * k1]
    
    # Find roots of the polynomial
    r = np.real(np.roots(g))
    
    # Check each point
    s = [t**2 / (1 + f2**2 * t**2) + (c * t + d)**2 / ((a * t + b)**2 + \
         f1**2 * (c * t + d)**2) for t in r]
         
    # Add value at asymptotic point
    s.append(1. / f2**2 + c**2 / (a**2 + f1**2 * c**2))
    
    # Check two possible cases
    i_min = np.argmin(s)
    if i_min < r.size:
        # Not point at infinity
        tmin = r[i_min]
        l1 = np.array([-f1 * (c * tmin + d), a * tmin + b, c * tmin + d])
        l2 = np.array([tmin * f2, 1, -tmin])
    else:
        # Special case: tmin = tinf
        l1 = np.array([-f1 * c, a, c])
        l2 = np.array([f2, 0., -1.])
    
    # Find closest points to origin
    find_closest = lambda l: np.array([-l[0] * l[2], 
                                       -l[1] * l[2], 
                                       l[0]**2 + l[1]**2]).reshape(-1,1)
    x1new = find_closest(l1)
    x2new = find_closest(l2)
    
    # Transfer back to original coordinate system
    x1new = np.dot(T1, np.dot(R1.T, x1new))
    x2new = np.dot(T2, np.dot(R2.T, x2new))

    # Find 3D point with linear method on new coordinates
    X = triangulate_linear(C1, C2, x1new, x2new)
    
    return X


def triangulate_linear(C1, C2, x1, x2):
    """Linear trinagulation of 3D point
    
    Parameters
    ------------------
    C1 : (3, 4) array
        First camera
    C2 : (3, 4) array
        Second camera
    x1 : (2,) array
        Image coordinates in first camera
    x2 : (2,) array
        Image coordinates in second camera
    
    Returns
    ------------------
    X : (3, 1) array
        The triangulated 3D point
    """
    if x1.shape[0] == 2:
        x1 = homog(x1)
        x2 = homog(x2)
    M = np.vstack([np.dot(cross_matrix(x1), C1),
                  np.dot(cross_matrix(x2), C2)])
    U, s, V = np.linalg.svd(M)
    X = V[-1,:]
    return X[:3] / X[-1]                 


def fmatrix_epipoles(F):
    """Epipoles of a fundamental matrix
    
    Parameters
    -------------------
    F : (3,3) array
        Fundamental matrix
    
    Returns
    -------------------
    e1 : (2,1) array
        Epipole 1
    e2 : (2,1) array
        Epipole 2
    """
    U, s, V = np.linalg.svd(F)
    e1 = U[:,-1]
    e2 = V[-1,:]
    
    e1 /= e1[-1]
    e2 /= e2[-1]
    
    return e1[:2], e2[:2]


def joint_min(A):
    """Joint minimum of a matrix
    
    This returns a list of all elements a_ij
    where a_ij is the minimum value of both row i
    and column j.
    
    Parameters
    --------------------
    A : (M, N) array
        A "match matrix" of values
    
    Returns
    --------------------
    vals : list
        List of minimum values such that vals[k] = A[ri[k], ci[k]]
    
    ri : (K, ) array
        Row coordinates for the found elements
    ci : (K, ) array
        Column coordinates for the found elements
    """
    col_mins = np.argmin(A, axis=0)
    row_mins = np.argmin(A, axis=1)
    
    ri = []
    ci = []
    for col, row in enumerate(col_mins):
        if row_mins[row] == col:
           ri.append(row)
           ci.append(col)
           
    vals = [A[i,j] for i, j in zip(ri, ci)] 
    
    return np.array(vals), np.array(ri), np.array(ci)


def cut_out_rois(image, col_indices, row_indices, roi_size):
    """Cut out regions of interest from an image
    
    Parameters
    -----------------
    image : (M, N) array
        Grayscale image
    col_indices : list or 1-D array
        ROI center x-coordinates
    row_indices : list or 1-D array
        ROI center y-coordinates
    roi_size : int
        Side of the ROI. Must be odd.
    
    Returns
    -----------------
    rois : list
        List of regions of interest (roi_size x roi_size arrays)
    
    """
    if not roi_size % 2 == 1:
        raise ValueError('ROI size must be odd')
    
    m = (roi_size-1)//2
    R,C = image.shape
    row_indices, col_indices = np.array(row_indices), np.array(col_indices)

    if np.any(R-m <= row_indices):
        raise ValueError('Row index to high, could not construct ROI')
    elif np.any(m > row_indices):
        raise ValueError('Row index to low, could not construct ROI')
    elif np.any(C-m <= col_indices):
        raise ValueError('Col index to high, could not construct ROI')
    elif np.any(m > col_indices):
        raise ValueError('Col index to low, could not construct ROI')

    rois = [image[r-m:r+m+1, c-m:c+m+1] for r,c in zip(row_indices, col_indices)]
    return rois


def plot_eplines(F, pts, imsize, **plot_kwargs):
    """Plot epipolar lines
    
    Given two images img1 and img2, with corresponding points p1, and p2
    this function can be used to plot the epipolar lines in either image 1 or image 2.
    
    To plot lines in image 1:
    
    >>> plot_eplines(F, y2, img1.shape)
    
    To plot lines image 2:
    
    >>> plot_eplines(F.T, y1, img2.shape)
    
    Parameters
    ------------------
    F : (3, 3) array
        The fundamental matrix that relates points as y1.T @ F y2 = 0
    pts : (2, N) array
        Points in image 2 (y2 above)
    imsize : tuple
        Tuple (width, height) that defines the image size of image 1
    plot_kwargs: dict (optional)
        Parameters passed to the matplotlib plot() command to draw each line
    
    """
    h, w = imsize[:2]
    
    # Image border lines
    l_left = np.array([1, 0, 0])
    l_right = np.array([1, 0, -w])
    l_bottom = np.array([0, 1, -h])
    l_top = np.array([0, 1, 0])
    
    margin = 0.5
    
    for i, pp in enumerate(pts.T):
        l = F.dot(homog(pp)) # Epipolar line
        endpoints = []
        for ll, name, limit in ((l_top, "top", h), (l_bottom, "bottom", h), 
                                (l_left, "left", w), (l_right, "right", w)):
            if len(endpoints) == 2:
                break

            p = np.cross(l, ll)  #  Epipolar line and image border intersection
            if np.isclose(p[2], 0):
                #print '- Failed infinity'
                continue # Point at infinity, try other axis
                
            x, y = p[:2] / p[2]
            if not (-margin <= x <= w+margin and -margin <= y <= h+margin):
                #print '- Failed limits'
                continue # Did not intersect axis, try other axis
            
            endpoints.append((x, y))
                
        if len(endpoints) == 2:
            a, b = endpoints
            plt.plot((a[0], b[0]), (a[1], b[1]), **plot_kwargs)
        else:
            warnings.warn("Failed to draw epipolar line for pts[{:d}]={}".format(
                            i, pp))
    plt.axis([0, w, 0, h])
    plt.gca().invert_yaxis()


def show_corresp(img1, img2, p1, p2, vertical=True):
    """Draw point correspondences
    
    Draws lines between corresponding points in two images.
    Hovering over a line highlights that line.
    
    Note that you need to call plt.show()
    after calling this function to view the results.
    
    
    Parameters
    ---------------
    img1 : (M, N) array
        First image
    img2 : (M, N) array
        Second image
    p1 : (2, K) array
        Points in first image
    p2 : (2, K) array
        Points in second image
    vertical: bool
        Controls layout of the images
        
    Returns
    ------------
    fig : Figure
        The drawn figure
    """
    assert p1.shape == p2.shape
    
    fig = plt.figure()
    plt.gray()
    rows, cols = (2,1) if vertical else (1,2)
    ax_left = fig.add_subplot(rows, cols, 1)
    ax_right = fig.add_subplot(rows, cols, 2)
    
    imshow_args = {'interpolation' : 'nearest'}
    im_left_artist = ax_left.imshow(img1, **imshow_args)
    im_right_artist = ax_right.imshow(img2, **imshow_args)
    
    connection_patches = []
    
    corr_data = {
         'active' : None,
         'patches' : connection_patches
    }
    
    def hover_cp(event):
        if corr_data['active'] is not None:
            if corr_data['active'].contains(event, radius=10)[0] == True:
                return
            else:
                plt.setp(corr_data['active'], color='b')
                plt.draw()
                corr_data['active'] = None
                
        for cp in corr_data['patches']:
            contained, cdict = cp.contains(event, radius=10)
            if contained == True:
                corr_data['active'] = cp
                plt.setp(cp, color='r')
                break
        plt.draw()
        
    for (xyA, xyB) in zip(p1.T, p2.T):
        cp = ConnectionPatch(xyA = xyB, xyB = xyA,
               coordsA='data', coordsB='data',
               axesA=ax_right, axesB=ax_left,
               arrowstyle='-', color='b')
        connection_patches.append(cp)
        ax_right.add_artist(cp)

    ax_right.set_zorder(ax_left.get_zorder() + 1)
    ax_left.plot(p1[0], p1[1], 'o')
    ax_right.plot(p2[0], p2[1], 'o')

    for im, ax in ((img1, ax_left), (img2, ax_right)):
        ax.set_xlim(0, im.shape[1]-1)
        ax.set_ylim(im.shape[0]-1, 0)

    
    fig.canvas.mpl_connect('motion_notify_event', hover_cp)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                        wspace=0.05, hspace=0.05)
                        
    return fig


def print_sparselm_info(info):
    """Human readable information from sparselm results"""
    d = {
        'e_initial' : info[0],
        'e_final' : info[1],
        'dp' : info[3],
        'niter' : int(info[5]),
        'reason_num' : int(info[6]),
        'nfev' : int(info[7]),
        'njac' : int(info[8]),
        'nlinsys' : int(info[9])
    }
    
    d['reason_str'] = {
        1 : 'stopped by small gradient J^T e',
        2 : 'stopped by small dp',
        3 : 'stopped by itmax',
        4 : 'singular matrix. Restart from current p with increased mu',
        5 : 'too many failed attempts to increase damping. Restart with increased mu',
        6 : 'stopped by small ||e||_2',
        7 : 'stopped by invalid (i.e. NaN or Inf) "func" values. User error'
    }.get(d['reason_num'], 'Unknown reason')
    
    d['e_percent'] = 100. * d['e_final'] / d['e_initial']
    
    print("""Optimization results
------------------------------
iterations:        {niter:d}
func. eval.:       {nfev:d}
initial residual:  {e_initial:.3e}
final residual:    {e_final:.3e}  ({e_percent:.3f}%)

Reason: ({reason_num}) {reason_str}""".format(**d))


def imshow(image, **kwargs):
    """Wrapper for matplotlibs imshow to turn off interpolation
    
    See documentation for matplotlib.pyplot.imshow.
    """
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'none'
    plt.imshow(image, **kwargs)
