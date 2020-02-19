import numpy as np
import cv2
from matchPics import matchPics
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from opts import get_opts

opts = get_opts()

# Q2.1.6
# Read the image and convert to grayscale, if necessary

cv_cover = cv2.imread('../data/cv_cover.jpg')

matches_count = []

for i in range(36):
    # Rotate Image

    angle_rotation = i * 10
    rotated_image = ndimage.rotate(cv_cover, angle_rotation, reshape=False)
    matches, locs1, locs2 = matchPics(cv_cover, rotated_image, opts)
    matches_count.append(len(matches))


# Compute features, descriptors and Match features


# Update histogram
plt.bar(np.arange(36), matches_count)
# Display histogram
plt.show()
