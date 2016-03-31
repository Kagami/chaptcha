#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import numpy as np
import cv2


CAPTCHA_WIDTH = 220
CAPTCHA_HEIGHT = 80
CH_WIDTH = 22
CH_HEIGHT = 44


def preprocess(img):
    # Remove noise.
    img = cv2.fastNlMeansDenoising(img, None, 65, 5, 21)
    img = cv2.threshold(img, 128, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Remove lines.
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100,
                            minLineLength=100, maxLineGap=100)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), 0, 2)
    return img


def split(img):
    def find_filled_row(rows):
        for y, row in enumerate(rows):
            dots = np.sum(row) // 255
            if dots > THRESHOLD:
                return y
        assert(False)

    def pad_ch(ch):
        pad_w = CH_WIDTH - len(ch.T)
        assert(pad_w >= 0)
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1
        pad_h = CH_HEIGHT - len(ch)
        assert(pad_h >= 0)
        pad_h1 = pad_h // 2
        pad_h2 = pad_h - pad_h1
        return np.pad(ch, ((pad_h1, pad_h2), (pad_w1, pad_w2)), 'constant')

    THRESHOLD = 3
    DELTA = 8

    # Search blank intervals.
    dots_per_col = np.apply_along_axis(lambda row: np.sum(row) // 255, 0, img)
    blanks = []
    was_blank = False
    prev = 0
    x = 0
    while x < CAPTCHA_WIDTH:
        if dots_per_col[x] > THRESHOLD:
            if was_blank:
                if prev:
                    blanks.append((prev, x - prev))
                x += DELTA
                was_blank = False
        elif not was_blank:
            was_blank = True
            prev = x
        x += 1
    blanks = sorted(blanks, key=lambda e: e[1])[:5]
    blanks = sorted(blanks, key=lambda e: e[0])
    # Add last (imaginary) blank to simplify following loop.
    blanks.append((prev if was_blank else CAPTCHA_WIDTH, 0))

    # Get chars.
    chars = []
    prev = 0
    widest = 0, 0
    for i, (x, _) in enumerate(blanks):
        ch = img[:CAPTCHA_HEIGHT, prev:x]
        prev = x
        x1 = find_filled_row(ch.T)
        x2 = len(ch.T) - find_filled_row(ch.T[::-1])
        width = x2 - x1
        # Don't allow more than CH_WIDTH * 2.
        extra_w = width - CH_WIDTH * 2
        extra_w1 = extra_w // 2
        extra_w2 = extra_w - extra_w1
        x1 = max(x1, x1 + extra_w1)
        x2 = min(x2, x2 - extra_w2)
        y2 = CAPTCHA_HEIGHT - find_filled_row(ch[::-1])
        y1 = max(0, y2 - CH_HEIGHT)
        ch = ch[y1:y2, x1:x2]
        chars.append(ch)
        if width > widest[0]:
            widest = x2 - x1, i

    # Fit chars into char box.
    chars2 = []
    for i, ch in enumerate(chars):
        widest_w, widest_i = widest
        height = len(ch)
        # Split glued chars.
        if len(chars) < 6 and i == widest_i:
            ch1 = ch[0:height, 0:widest_w // 2]
            ch2 = ch[0:height, widest_w // 2:widest_w]
            chars2.append(pad_ch(ch1))
            chars2.append(pad_ch(ch2))
        else:
            ch = ch[0:height, 0:CH_WIDTH]
            chars2.append(pad_ch(ch))

    assert(len(chars2) == 6)
    return chars2


def show(img):
    cv2.imshow('opencv-result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', dest='infile', metavar='infile',
        help='input file')
    parser.add_argument(
        '-c', dest='crop', action='store_true',
        help='show cropped chars')
    parser.add_argument(
        'mode', choices=['show'],
        help='operational mode')
    opts = parser.parse_args(sys.argv[1:])
    if opts.mode == 'show':
        if opts.infile is None:
            parser.error('specify input file')
        img = cv2.imread(opts.infile, 0)
        assert(img is not None)
        img = preprocess(img)
        if opts.crop:
            img = np.concatenate(split(img), axis=1)
        show(img)


if __name__ == '__main__':
    main()
