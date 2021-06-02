import argparse
import cv2 as cv

from tools import detect_lines, reduce_objects, get_music, play
from detection import get_musical_objects

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argument_parser.add_argument('-i', '--input-image', help='Filepath to input image',
                                 dest='i', default='data/simple/04.png', metavar='')
    argument_parser.add_argument('-m', '--model', help='Filepath to pretrained object detection model',
                                 dest='m', default='resources/model.pb', metavar='')
    argument_parser.add_argument('-s', '--save', help='Save output music as an .MP3 file',
                                 dest='s', action="store_true")
    args = argument_parser.parse_args()

    image = cv.imread(args.i, 0)
    lines = detect_lines(image)

    objects = get_musical_objects(image_path=args.i, model_path=args.m)

    notes, groups = reduce_objects(objects, lines)

    music = get_music(notes, lines)
    
    play(music, save=args.s)
