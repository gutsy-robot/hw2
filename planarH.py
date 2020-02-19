import numpy as np
import cv2
from numpy import linalg as LA


def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points

    # get four corresponding points from x1 and x2.
    assert len(x1) == len(x2), "inputs are not of same shape"
    assert len(x1) >= 4 and len(x2) >= 4, "at least 4 points required to determine H."
    indices = np.random.randint(len(x1), size=4)
    a = None
    for ind in indices:
        x, y = x2[ind]
        u, v = x1[ind]

        m = np.array([[x, y, 1, 0, 0, 0, -u * x, -u * y, -u],
                      [0, 0, 0, x, y, 1, -v * x, -v * y, -v]])
        if a is None:
            a = m
        else:
            a = np.vstack((a, m))

    U, S, V = np.linalg.svd(a)
    H2to1 = V.T[:, -1].reshape([3, 3])

    # print("mtrix A is: ", a)
    # w, vec = LA.eig(np.matmul(a.T, a))
    #
    # print("w is: ", w)
    # # print("vec is: ", vec)
    # min_eig_index = np.argmin(w)
    # print("index is: ", min_eig_index)
    # print(vec[min_eig_index])
    # print(vec[-1])
    # H2to1 = vec[min_eig_index]
    #
    # H2to1 = H2to1.reshape(3, 3)

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points

    # Shift the origin of the points to the centroid

    print("shape of inputs in norm: ", x1.shape, x2.shape)

    m1, m2 = x1.mean(0),  x2.mean(0)
    print(m1, m2)

    x1 = x1 - m1
    x2 = x2 - m2

    print("shape after subtraction is: ", x1.shape, x2.shape)

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max1 = np.max(LA.norm(x1, axis=1))
    max2 = np.max(LA.norm(x2, axis=1))

    s1 = 1 / (max1 / 2 ** 0.5)
    s2 = 1 / (max2 / 2 ** 0.5)

    x1 *= s1
    x2 *= s2

    # Similarity transform 1
    T1 = np.array([[s1, 0, -s1 * m1[0]], [0, s1, -s1 * m1[1]], [0, 0, 1]])

    # Similarity transform 2
    T2 = np.array([[s2, 0, -s2 * m2[0]], [0, s2, -s2 * m2[1]], [0, 0, 1]])

    # Compute homography
    h = computeH(x1, x2)

    # Denormalization
    print("shape pf T1 is; ", T1.shape)
    print("shape of T2 is: ", T2.shape)
    print("shape of h is: ", h.shape)
    H2to1 = np.linalg.inv(T1).dot(h).dot(T2)

    return H2to1


def computeH_ransac(locs1, locs2, opts):

    # Note: H can be further improved by finding H by fitting all the inlier points.

    # Q2.2.3
    # Compute the best fitting homography given a list of matching points

    print("shape of locs: ", locs1.shape, locs2.shape)
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol  # the tolerance value for considering a point to be an inlier

    # indices = np.random.randint(len(locs1), size=4)
    locs1_h = np.hstack((locs1, np.ones((len(locs1), 1))))
    locs2_h = np.hstack((locs2, np.ones((len(locs2), 1))))
    best_h, inliers = None, None
    for i in range(0, max_iters):
        h = computeH_norm(locs1, locs2)
        locs2_mapped = np.matmul(h, locs2_h.T)
        locs2_mapped /= locs2_mapped[2, :]
        locs2_mapped[2, :] = np.ones(locs2_mapped.shape[1])

        error = locs1_h.T - locs2_mapped
        # error_euc = error / error[2, :]
        # error_euc[2, :] = np.zeros(error_euc.shape[1])
        error_final = np.sum(error ** 2, axis=1)
        err = error_final <= inlier_tol
        inl = err.astype(int)

        if best_h is None and inliers is None:
           best_h = h
           inliers = inl

        else:
            if np.sum(inl == 1) > np.sum(inliers == 1):
                best_h = h
                inliers = inl

    bestH2to1 = best_h

    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template

    # Warp mask by appropriate homography

    # Warp template by appropriate homography

    # Use mask to combine the warped template and the image


    return composite_img
