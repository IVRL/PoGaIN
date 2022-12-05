"""
PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

Authors: Nicolas Bähler, Majed El Helou, Étienne Objois, Kaan Okumuş, and Sabine
Süsstrunk, Fellow, IEEE.

This file contains utility methods.
"""

import numpy as np
from PIL import Image
from scipy.stats import norm, poisson


def load_image(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    # Load the image and convert into numpy array
    img = Image.open(path)
    array = np.asarray(img)

    if len(array.shape) == 3:
        array = np.mean(array, axis=2)

    array = array / 255

    return (array.flatten(), array.shape)


def add_noise(x: np.ndarray, a: float, b: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    n = len(x)

    poisson_noise = poisson.rvs(mu=a * x, size=n)
    gaussian_noise = norm.rvs(scale=b, size=n)

    return poisson_noise / a + gaussian_noise
