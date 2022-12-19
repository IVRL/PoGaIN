"""
PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

Authors: Nicolas Bähler, Majed El Helou, Étienne Objois, Kaan Okumuş, and Sabine
Süsstrunk, Fellow, IEEE.
"""

import os
import shutil
import sys
from pathlib import Path

from scipy.io import loadmat, savemat

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import utils

_current_a = None
_images_path = None
_images = None


def set_current_a(current_a):
    global _current_a
    _current_a = current_a


def set_images(images_path, images):
    global _images_path
    global _images

    _images_path = images_path
    _images = images


def prepare_data(a, b, seed):
    res_path = os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    y_path = os.path.join(os.path.dirname(__file__), "data", _current_a)

    if os.path.exists(y_path):
        shutil.rmtree(y_path)
    os.makedirs(y_path)

    res_path = os.path.join(res_path, _current_a)

    if os.path.exists(res_path):
        shutil.rmtree(res_path)
    os.makedirs(res_path)

    for image in _images:
        x, shape = utils.load_image(os.path.join(_images_path, image))
        y = utils.add_noise(x, a, b, seed=seed)
        f = os.path.join(y_path, f"{os.path.splitext(image)[0]}.mat")
        savemat(f, {"data": y.reshape(shape)})


def delete_prepared_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", _current_a)

    if os.path.exists(data_path):
        shutil.rmtree(data_path)


def get_results():
    data_path = os.path.join(os.path.dirname(__file__), "results", _current_a)

    file_names = []
    with open(f"{data_path}/FileNames.txt") as file:
        file_names.extend(line.rstrip() for line in file)

    NoiseAB = loadmat(f"{data_path}/NoiseAB.mat")["NoiseAB"]
    a = NoiseAB[:, 0]
    b = NoiseAB[:, 1]

    return file_names, a, b
