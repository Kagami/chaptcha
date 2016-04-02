# chaptcha

Break CAPTCHA at 2ch.hk using OpenCV and FANN. Just for fun (UWBFTP).

## Demo

Demo backend is available at `ch.genshiken.org`, [install userscript](https://raw.githubusercontent.com/Kagami/chaptcha/master/chaptcha.user.js) to try it out (not tested for compatibility with Dollchan Extension Tools).

![](https://raw.githubusercontent.com/Kagami/chaptcha/assets/vis.png)
![](https://raw.githubusercontent.com/Kagami/chaptcha/assets/cap.gif)

## Requirements

* [Python](https://www.python.org/) 2.7+ or 3.2+
* [NumPy](http://www.numpy.org/) 1.7+
* [OpenCV](http://opencv.org/) 2.4+ with Python bindings
* [FANN](http://leenissen.dk/fann/wp/) 2.1+ with Python bindings
* [requests](http://python-requests.org/) 2+
* [bottle](http://bottlepy.org/) 0.10+

## Usage

```bash
# Visualize preprocess/segmentation steps
python chaptcha.py vis -i captcha.png
# Collect training data
python chaptcha.py collect -o captchas/ -k ag.txt
# Train neural network
python chaptcha.py train -i captchas/ -o my.net
# Recognize CAPTCHA
python chaptcha.py ocr -i captcha.png -n my.net
# Host OCR backend (for chaptcha.user.js)
python chaptcha.py serve -n my.net
```

## Links

* [Image Denoising](http://docs.opencv.org/3.1.0/d5/d69/tutorial_py_non_local_means.html)
* [Hough Line Transform](http://docs.opencv.org/3.1.0/d6/d10/tutorial_py_houghlines.html)
* [Python FANN tutorial](http://jansipke.nl/using-fann-with-python/)
* [Break simple CAPTCHA](https://habrahabr.ru/post/63854/)
* [Break another simple CAPTCHA](http://cybern.ru/raspoznavanie-kapchi-captcha.html)
* [Break ifolder CAPTCHA](https://geektimes.ru/post/67194/)
* [Break rzd CAPTCHA](https://toster.ru/q/216509)

## License

[CC0.](COPYING)
