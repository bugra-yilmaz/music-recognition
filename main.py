from tools import *

if __name__ == "__main__":
    image = cv.imread("data/simple/04.png", 0)

    image = binarize(image)

    angle, image = rotate(image)

    rows = detect_lines(image)

    cv.imwrite("output.png", image)
