import cv2 as cv
import numpy as np
from scipy.ndimage import interpolation as inter


# Applies Gaussian blur and Otsu's binarization method to the given grayscale image
def binarize(image):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    _, binarized = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return binarized


# Corrects the skew in the given image by rotating it
def rotate(image, delta=2, limit=20):

    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv.warpAffine(image, m, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    return best_angle, rotated
