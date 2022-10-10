"""
PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

Authors: Nicolas Bähler, Majed El Helou, Étienne Objois, Kaan Okumuş, and Sabine
Süsstrunk, Fellow, IEEE.
"""

import numpy as np
from scipy.stats import poisson, norm
from PIL import Image


def load_image(path):
    # load the image and convert into numpy array
    img = Image.open(path)
    array = np.asarray(img)

    if len(array.shape) == 3:
        array = np.mean(array, axis=2)

    array = array / 255

    return (array.flatten(), array.shape)


def add_noise(x, a, b, seed):
    np.random.seed(seed)
    n = len(x)

    poisson_noise = poisson.rvs(mu=a * x, size=n)
    gaussian_noise = norm.rvs(scale=b, size=n)

    return poisson_noise / a + gaussian_noise
