import cv2 as cv

from tools import detect_lines, reduce_objects
from detection import get_musical_objects

if __name__ == '__main__':
    image_path = 'data/simple/02.png'

    image = cv.imread(image_path, 0)
    lines = detect_lines(image)

    objects = get_musical_objects(image_path)

    notes = reduce_objects(objects, lines)
