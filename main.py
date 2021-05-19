from tools import *
from detection import get_musical_objects

if __name__ == '__main__':
    image = cv.imread('data/simple/02.png', 0)

    image = binarize(image)
    image = rotate(image)
    lines = detect_lines(image)
    cv.imwrite('temp.png', image)

    objects = get_musical_objects('temp.png')

    notes = reduce_objects(objects)

    music_input = classify_notes(notes, lines)

    play(music_input)
