import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, computeH, computeH_norm, compositeH
from numpy import linalg as LA


# Import necessary functions

# Write script for Q2.2.4

opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

print("shapes are: ", cv_cover.shape, hp_cover.shape)

h, w = cv_cover.shape[0], cv_cover.shape[1]

resized = np.zeros(cv_cover.shape)
resized1 = cv2.resize(hp_cover[:, :, 0], (w, h))
resized2 = cv2.resize(hp_cover[:, :, 1], (w, h))
resized3 = cv2.resize(hp_cover[:, :, 2], (w, h))

resized[:, :, 0] = resized1
resized[:, :, 1] = resized2
resized[:, :, 2] = resized3

print("shapes are: ", cv_cover.shape, resized.shape)
# cv2.imwrite('resized.png', resized)
# rotated_image = ndimage.rotate(cv_desk, 40, reshape=False)

matches, l1, l2 = matchPics(cv_desk, cv_cover, opts)

# l1 = np.random.random((4, 2))
# print("first: ", computeH_norm(l1, l1))

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

print("h is: ", h)

im = cv2.warpPerspective(resized.swapaxes(0, 1), h, (cv_desk.shape[0], cv_desk.shape[1])).swapaxes(0, 1)
cv2.imwrite('perspective3.png', im)
print("shape of im is: ", im.shape, cv_desk.shape)

# print("shape of im is: ", im.shape)
final_image = compositeH(h, cv_desk, resized)
cv2.imwrite('final.png', final_image)
# cv2.warpPerspective(hp_cover, h, (cv_desk.shape[0], cv_desk.shape[1]))
# print("shape of l1 and l2: ", l1.shape, l2.shape)

# im = cv2.warpPerspective(cv_desk.swapaxes(0, 1), LA.inv(h), (cv_cover.shape[1], cv_cover.shape[0])).swapaxes(0, 1)
# cv2.imwrite('perspective4.png', im)