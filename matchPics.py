import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection, plotMatches


def matchPics(I1, I2, opts):
    # I1, I2 : Images to match
    # opts: input opts
    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'

    # Convert Images to GrayScale
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # print("shape of I1 is: ", I1.shape)
    # print("shape of I2 is: ", I2.shape)
    # print("converted to gray")

    # Detect Features in Both Images
    locs1 = corner_detection(I1, sigma)
    locs2 = corner_detection(I2, sigma)

    # print("shape of features1 is: ", locs1.shape)
    # print("shape of features2 is: ", locs2.shape)
    # print("corner detection done")

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1, locs1)
    desc2, locs2 = computeBrief(I2, locs2)

    # print("shape of desc1 is: ", desc1.shape)
    # print("shape of desc2 is: ", desc2.shape)
    # print("shape of locs1 is: ", locs1.shape)
    # print("shape of locs2 is: ", locs2.shape)
    # print("compute brief done")

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    # plotMatches(I2, I1, matches, locs1, locs2)
    # print("shape of matches is; ", matches.shape)
    # print("shape of locs1 is: ", locs1.shape)
    # print("shape of locs2 is: ", locs2.shape)
    return matches, locs1, locs2
