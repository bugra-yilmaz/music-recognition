from tools import *

if __name__ == "__main__":
    image = cv.imread("data/simple/04.png", 0)
    image = binarize(image)
    angle, image = rotate(image)
    rows = detect_lines(image)
    template = cv.imread("data/templates/eight.png", 0)
    image = normxcorr2(template, image)
    image = scale(image)
    cv.imwrite("output.png", image)
