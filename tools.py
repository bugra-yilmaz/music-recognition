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

    lines = find_peaks(summed, maximum / 2)[0]

    line_groups = list()
    for i in range(5, len(lines)+1, 5):
        line_groups.append(lines[i-5:i])

    return line_groups


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


def assign_groups(objects, lines):
    line_centers = [sum(line_group) / len(line_group) for line_group in lines]
    indexes = list(range(len(line_centers)))

    groups = list()
    for o in objects:
        y_center = o[0][1]
        group = min(indexes, key=lambda i: abs(line_centers[i] - y_center))
        groups.append(group)

    return groups


def is_quarter(note, group, beams):
    x_center = note[0][0]
    for beam in beams:
        if beam[1] != group:
            continue

        x_min, x_max = beam[0]
        if x_min < x_center < x_max:
            return True

    return False


# Reduces recognized musical objects to musical notes
def reduce_objects(objects, lines):
    groups = assign_groups(objects, lines)

    beams = list()
    for i, o in enumerate(objects):
        if o[1] != 'beam':
            continue

        x_min, x_max = o[0][0]
        offset = (x_max - x_min) * 0.2
        x_min = x_min - offset
        x_max = x_max + offset

        beam = (x_min, x_max), groups[i]
        beams.append(beam)

    notes = list()
    for i, o in enumerate(objects):
        group = groups[i]
        if o[1] == 'notehead-empty':
            note = (o[0][0], o[0][1], group, 2)
            notes.append(note)

        elif o[1] == 'notehead-full':
            if is_quarter(o, group, beams):
                note = (o[0][0], o[0][1], group, 0)
            else:
                note = (o[0][0], o[0][1], group, 1)
            notes.append(note)

    notes = sorted(notes, key=lambda x: (x[2], x[0]))

    return notes


# Classifies musical notes according to previously detected staff lines
def classify_notes(notes, lines):
    pass


# Produces the music output from the input
def play(music_input):
    pass
