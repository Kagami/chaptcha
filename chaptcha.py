#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import sys
import time
import argparse
import numpy as np
import cv2
from fann2 import libfann


CAPTCHA_WIDTH = 220
CAPTCHA_HEIGHT = 80
CH_WIDTH = 22
CH_HEIGHT = 44


def get_image(fpath):
    img = cv2.imread(fpath, 0)
    assert(img is not None)
    return img


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
        raise Exception('cannot find filled row')

    def pad_ch(ch):
        pad_w = CH_WIDTH - len(ch.T)
        assert pad_w >= 0, 'bad char width'
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1
        pad_h = CH_HEIGHT - len(ch)
        assert pad_h >= 0, 'bad char height'
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

    assert len(chars2) == 6, 'bad number of chars'
    return chars2


def show(img):
    cv2.imshow('opencv-result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def report(line='', progress=False):
    if progress:
        line = '\033[1A\033[K' + line
    line += '\n'
    sys.stderr.write(line)


def train(captchas_dir):
    CONNECTION_RATE = 1
    LEARNING_RATE = 0.7
    NUM_INPUT = CH_WIDTH * CH_HEIGHT
    NUM_NEURONS_HIDDEN = 144
    NUM_OUTPUT = 10
    ann = libfann.neural_net()
    ann.create_sparse_array(CONNECTION_RATE,
                            (NUM_INPUT, NUM_NEURONS_HIDDEN, NUM_OUTPUT))
    ann.set_learning_rate(LEARNING_RATE)
    ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

    start = time.time()
    captchas_dir = os.path.abspath(captchas_dir)
    captchas = os.listdir(captchas_dir)
    succeed = 0
    report()
    for i, name in enumerate(captchas):
        answers = re.match(r'(\d{6})\.png$', name)
        if not answers:
            continue
        answers = answers.group(1)
        fpath = os.path.join(captchas_dir, name)
        try:
            img = get_image(fpath)
            img = preprocess(img)
            ch_imgs = split(img)
            for ch_img, answer in zip(ch_imgs, answers):
                ann.train(img2data(ch_img), make_answer(answer))
        except Exception as exc:
            report('Error occured while processing {}: {}'.format(name, exc))
            report()
        else:
            succeed += 1
            report('{}/{}'.format(i + 1, len(captchas)), progress=True)
    runtime = time.time() - start
    report('Done training on {}/{} captchas in {:.3f} seconds'.format(
           succeed, len(captchas), runtime))
    return ann


def img2data(img):
    data = img.flatten() & 1
    assert len(data) == CH_WIDTH * CH_HEIGHT, 'bad data size'
    return data


def make_answer(digit):
    digit = int(digit)
    answer = [-1] * 10
    answer[digit] = 1
    return answer


def get_network(fpath):
    ann = libfann.neural_net()
    ann.create_from_file(fpath)
    return ann


def ocr(ann, img):
    def find_answer(ch_img):
        data = img2data(ch_img)
        res = ann.run(data)
        return str(res.index(max(res)))

    img = preprocess(img)
    return ''.join(map(find_answer, split(img)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', dest='infile', metavar='infile',
        help='input file/directory')
    parser.add_argument(
        '-o', dest='outfile', metavar='outfile',
        help='output file/directory')
    parser.add_argument(
        '-c', dest='crop', action='store_true',
        help='show cropped chars')
    parser.add_argument(
        '-n', dest='netfile', metavar='netfile',
        help='neural network')
    parser.add_argument(
        'mode', choices=['show', 'train', 'ocr'],
        help='operational mode')
    opts = parser.parse_args(sys.argv[1:])
    if opts.mode == 'show':
        if opts.infile is None:
            parser.error('specify input captcha')
        img = get_image(opts.infile)
        img = preprocess(img)
        if opts.crop:
            img = np.concatenate(split(img), axis=1)
        show(img)
    elif opts.mode == 'train':
        if opts.infile is None:
            parser.error('specify input directory with captchas')
        if opts.outfile is None:
            parser.error('specify output file for network data')
        ann = train(opts.infile)
        ann.save(opts.outfile)
    elif opts.mode == 'ocr':
        if opts.infile is None:
            parser.error('specify input captcha')
        if opts.netfile is None:
            parser.error('specify network file')
        ann = get_network(opts.netfile)
        img = get_image(opts.infile)
        print(ocr(ann, img))


if __name__ == '__main__':
    main()
