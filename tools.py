import cv2 as cv
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import fftconvolve
from scipy.ndimage import interpolation


# Applies Gaussian blur and Otsu's binarization method to the given grayscale image
def binarize(image):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    _, binarized = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return binarized


# Corrects the skew in the given image by rotating it
def rotate(image, delta=2, limit=20):
    def determine_score(arr, angle):
        data = interpolation.rotate(arr, angle, reshape=False, order=0)
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

    return rotated


# Detects rows of staff lines in the given image
def detect_lines(image):
    inverse = cv.bitwise_not(image)
    summed = np.sum(inverse, axis=1)
    maximum = max(summed)

    rows = find_peaks(summed, maximum / 2)

    return rows[0]


# Removes staff lines from the image
def remove_lines(image, rows):
    length = len(image[0])
    for row in rows:
        for i in range(row-1, row+2):
            image[i] = np.asarray([255] * length)

    return image


# Normalized cross-correlation between a template image and an input image
def normxcorr2(template, image, mode="full"):
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
        np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


# Scales the given array to 0-255 range
def scale(corr):
    minimum = np.min(corr)
    maximum = np.max(corr)
    scalar = 255 / (maximum - minimum)

    scaled = np.round((corr - minimum) * scalar).astype(int)

    return scaled


# Reduces recognized musical objects to musical notes
def reduce_objects(objects):
    pass


# Classifies musical notes according to previously detected staff lines
def classify_notes(notes, lines):
    pass


# Produces the music output from the input
def play(music_input):
    pass
