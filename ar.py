import numpy as np
import cv2
from matchPics import matchPics
from planarH import computeH_ransac, computeH, computeH_norm, compositeH # Import necessary functions
from loadVid import loadVid
from opts import get_opts

# Write script for Q3.1

opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')

book_movie_path = "../data/book.mov"
panda_movie_path = "../data/ar_source.mov"

book_frames = loadVid(book_movie_path)
panda_frames = loadVid(panda_movie_path)

transformed_frames = []

for j in range(0, len(panda_frames[:20])):
    f = book_frames[j]
    p = panda_frames[j]

    h, w = cv_cover.shape[0], cv_cover.shape[1]

    resized = np.zeros(cv_cover.shape)
    resized1 = cv2.resize(p[:, :, 0], (w, h))
    resized2 = cv2.resize(p[:, :, 1], (w, h))
    resized3 = cv2.resize(p[:, :, 2], (w, h))

    resized[:, :, 0] = resized1
    resized[:, :, 1] = resized2
    resized[:, :, 2] = resized3

    matches, l1, l2 = matchPics(f, cv_cover, opts)
    matches1, matches2 = matches[:, 0], matches[:, 1]

    locs1, locs2 = [], []

    for i in range(0, len(matches1)):
        locs1.append(l1[matches1[i]])
        locs2.append(l2[matches2[i]])

    locs1 = np.array(locs1)
    locs2 = np.array(locs2)
    h, inliers = computeH_ransac(locs1, locs2, opts)

    final_image = compositeH(h, f, cv_cover)
    cv2.imwrite('results/' + str(j) + '.png', final_image)
