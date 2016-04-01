#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import sys
import time
import argparse
from datetime import datetime
import threading
import numpy as np
import cv2
from fann2 import libfann
import requests
import bottle


CAPTCHA_WIDTH = 220
CAPTCHA_HEIGHT = 80
CH_WIDTH = 22
CH_HEIGHT = 44


def check_image(img):
    assert img is not None, 'cannot read image'
    assert img.shape == (CAPTCHA_HEIGHT, CAPTCHA_WIDTH), 'bad image dimensions'


def get_image(fpath):
    img = cv2.imread(fpath, 0)
    check_image(img)
    return img


def decode_image(data):
    img = cv2.imdecode(data, 0)
    check_image(img)
    return img


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
    assert ann.create_from_file(fpath), 'cannot init network'
    return ann


def report(line='', progress=False):
    if progress:
        line = '\033[1A\033[K' + line
    line += '\n'
    sys.stderr.write(line)


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
        # Split glued chars.
        if len(chars) < 6 and i == widest_i:
            ch1 = ch[:, 0:widest_w // 2]
            ch2 = ch[:, widest_w // 2:widest_w]
            chars2.append(pad_ch(ch1))
            chars2.append(pad_ch(ch2))
        else:
            ch = ch[:, 0:CH_WIDTH]
            chars2.append(pad_ch(ch))

    assert len(chars2) == 6, 'bad number of chars'
    return chars2


def show(img):
    cv2.imshow('opencv-result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    succeed = 0
    captchas_dir = os.path.abspath(captchas_dir)
    captchas = os.listdir(captchas_dir)
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


def ocr(ann, img):
    def find_answer(ch_img):
        data = img2data(ch_img)
        res = ann.run(data)
        return str(res.index(max(res)))

    img = preprocess(img)
    return ''.join(map(find_answer, split(img)))


def antigate_ocr(api_key, data, timeout=90, ext='png',
                 is_numeric=True, min_len=6, max_len=6,
                 lock=None):
    def check_lock():
        if lock is None:
            return True
        else:
            return lock.is_set()

    FIRST_SLEEP = 7
    ATTEMPT_SLEEP = 2
    start = datetime.now()

    # Uploading captcha.
    fields = {'key': api_key, 'method': 'post'}
    if is_numeric:
        fields['numeric'] = '1'
    if min_len:
        fields['min_len'] = str(min_len)
    if max_len:
        fields['max_len'] = str(max_len)
    files = {'file': ('captcha.' + ext, data)}
    res = requests.post('http://anti-captcha.com/in.php',
                        data=fields, files=files).text
    if not res.startswith('OK|'):
        raise Exception(res)
    captcha_id = res[3:]
    # Getting captcha text.
    fields2 = {
        'key': api_key,
        'action': 'get',
        'id': captcha_id,
    }
    time.sleep(FIRST_SLEEP)
    while check_lock():
        res = requests.get('http://anti-captcha.com/res.php',
                           params=fields2).text
        if res.startswith('OK|'):
            return res[3:]
        elif res == 'CAPCHA_NOT_READY':
            if (datetime.now() - start).seconds >= timeout:
                raise Exception('getting captcha text timeout')
            time.sleep(ATTEMPT_SLEEP)
        else:
            raise Exception(res)
    raise Exception('antigate_ocr canceled')


def get_captcha():
    CAPTCHA_URL = 'https://2ch.hk/makaba/captcha.fcgi'
    CAPTCHA_FIELDS = {
        'type': '2chaptcha',
        'action': 'thread',
        'board': 's',
    }
    CHROME_UA = (
        'Mozilla/5.0 (Windows NT 6.1; WOW64) ' +
        'AppleWebKit/537.36 (KHTML, like Gecko) ' +
        'Chrome/48.0.2564.116 Safari/537.36'
    )
    CAPTCHA_HEADERS = {
        'User-Agent': CHROME_UA,
    }
    res = requests.get(CAPTCHA_URL, params=CAPTCHA_FIELDS,
                       headers=CAPTCHA_HEADERS).text
    if not res.startswith('CHECK\n'):
        raise Exception('bad answer on captcha request')
    captcha_id = res[6:]
    fields2 = {
        'type': '2chaptcha',
        'action': 'image',
        'id': captcha_id,
    }
    r = requests.get(CAPTCHA_URL, params=fields2, headers=CAPTCHA_HEADERS)
    if r.headers['Content-Type'] != 'image/png':
        raise Exception('bad captcha result')
    return r.content


def collect(lock, captchas_dir, tmp_path, api_key):
    while lock.is_set():
        try:
            data = get_captcha()
            answer = antigate_ocr(api_key, data, lock=lock)
            if not re.match(r'\d{6}$', answer):
                raise Exception('bad antigate answer {}'.format(answer))
            name = answer + '.png'
            fpath = os.path.join(captchas_dir, name)
            # In order to not leave partial files.
            open(tmp_path, 'wb').write(data)
            os.rename(tmp_path, fpath)
        except Exception as exc:
            report('Error occured while collecting: {}'.format(exc))
        else:
            report('Saved {}'.format(name))


def run_collect_threads(captchas_dir, api_key):
    NUM_THREADS = 10
    SPAWN_DELAY = 0.5

    threads = []
    lock = threading.Event()
    lock.set()
    captchas_dir = os.path.abspath(captchas_dir)

    for i in range(NUM_THREADS):
        tmp_path = os.path.join(captchas_dir, '.{}.tmp'.format(i))
        thread = threading.Thread(target=collect,
                                  args=(lock, captchas_dir, tmp_path, api_key))
        threads.append(thread)
        thread.start()
        time.sleep(SPAWN_DELAY)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        report('Closing threads')
        lock.clear()
        for thread in threads:
            thread.join()


@bottle.post('/ocr')
def serve():
    bottle.response.set_header('Access-Control-Allow-Origin', '*')
    try:
        fh = bottle.request.files['file'].file
    except Exception:
        bottle.abort(400, 'No file provided.')
    ann = bottle.request.app.ann
    img = decode_image(np.fromfile(fh, dtype=np.uint8))
    return ocr(ann, img)


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
        help='crop chars (for show)')
    parser.add_argument(
        '-n', dest='netfile', metavar='netfile',
        help='neural network (for ocr & serve)')
    parser.add_argument(
        '-k', dest='keyfile', metavar='keyfile',
        help='antigate key file (for collect)')
    parser.add_argument(
        '-p', dest='port', metavar='port',
        type=int, default=28228,
        help='listening port (for serve, default: %(default)s)')
    parser.add_argument(
        '-b', dest='host', metavar='host',
        default='127.0.0.1',
        help='listening address (for serve, default: %(default)s)')
    parser.add_argument(
        'mode', choices=['show', 'train', 'ocr', 'collect', 'serve'],
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
    elif opts.mode == 'collect':
        if opts.outfile is None:
            parser.error('specify output directory for captchas')
        if opts.keyfile is None:
            parser.error('specify antigate key file')
        api_key = open(opts.keyfile, 'r').read().strip()
        run_collect_threads(opts.outfile, api_key)
    elif opts.mode == 'serve':
        if opts.netfile is None:
            parser.error('specify network file')
        ann = get_network(opts.netfile)
        bottle.default_app().ann = ann
        bottle.run(host=opts.host, port=opts.port)


if __name__ == '__main__':
    main()
