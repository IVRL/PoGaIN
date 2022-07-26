import numpy as np
import os
from scipy.stats import poisson, norm
from PIL import Image
from pathlib import Path


def load_image(path=None, keep_shape=False):
    # load the image and convert into
    # numpy array
    if path is None:
        path = (
            f"{Path(os.path.abspath(os.path.dirname(__file__))).parent}/data/img3.jpeg"
        )

    img = Image.open(path)
    array = np.asarray(img)

    if len(array.shape) == 3:
        array = np.mean(array, axis=2)

    array = array / 255

    if keep_shape:
        return array

    return array.flatten(), array.shape


def add_noise(x, a, b, seed):
    np.random.seed(seed)
    n = len(x)
    poisson_noise = poisson.rvs(mu=a * x, size=n)
    gaussian_noise = norm.rvs(scale=b, size=n)
    return poisson_noise / a + gaussian_noise
