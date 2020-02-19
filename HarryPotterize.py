import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac

# Import necessary functions

# Write script for Q2.2.4

opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, l1, l2 = matchPics(cv_desk, cv_cover, opts)

matches1, matches2 = matches[:, 0], matches[:, 1]
locs1, locs2 = [], []
# locs1, locs2 = [], []

for i in range(0, len(matches1)):
    locs1.append(l1[matches1[i]])
    locs2.append(l2[matches2[i]])

locs1 = np.array(locs1)
locs2 = np.array(locs2)

print("shape of locs1 to ransac is: ", locs1.shape)
print("shape of locs2 to ransac is: ", locs2.shape)

h, inliers = computeH_ransac(locs1, locs2, opts)
print("shape of output h is: ", h.shape)
print("shape of inliers is: ", inliers.shape)

im = cv2.warpPerspective(hp_cover, h, (cv_desk.shape[1], cv_desk.shape[0]))
cv2.imwrite('perspective.png', im)
# cv2.warpPerspective(hp_cover, h, (cv_desk.shape[0], cv_desk.shape[1]))
# print("shape of l1 and l2: ", l1.shape, l2.shape)

