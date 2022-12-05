"""
PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

Authors: Nicolas Bähler, Majed El Helou, Étienne Objois, Kaan Okumuş, and Sabine
Süsstrunk, Fellow, IEEE.

This file contains the implementation of the baseline variance method, our
method (cumulant-based) and the log-likelihood function.
"""

import numpy as np
from scipy.stats import kstat, norm, poisson


def var(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Estimate a and b using our variance approach.

    Parameters
    ----------
    x : np.ndarray
        The ground truth noise-free image
    y : np.ndarray
        The noisy image

    Returns
    -------
    a : float or nan
        Estimation of a
    b : float or nan
        Estimation of b
    """
    # Create bins of pixel with the same value
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    # Compute the the difference squared
    x_unique, x_index, x_counts = np.unique(
        x_sorted, return_index=True, return_counts=True
    )
    splitted = np.split(y_sorted - x_sorted, x_index[1:])

    var = np.zeros(x_unique.size)
    for i, element in enumerate(splitted):
        var[i] = np.square(element).mean()

    # Formulate the problem as a linear matrix equation
    lhs = np.array([x_unique, np.ones(x_unique.size)]).T
    lhs_weighted = lhs * np.sqrt(x_counts[:, np.newaxis])
    rhs_weighted = var * np.sqrt(x_counts)

    # Compute the lest squares solution
    parameters, _, _, _ = np.linalg.lstsq(lhs_weighted, rhs_weighted, rcond=None)

    # b squared can't be negative
    if parameters[1] < 0:
        # Assuming b is 0 and redo the estimation for a with b==0
        lhs_weighted = np.array(lhs_weighted[:, 0])[:, None]
        parameters, _, _, _ = np.linalg.lstsq(lhs_weighted, rhs_weighted, rcond=None)

        b = 0
    else:
        b = np.sqrt(parameters[1])

    # a must be strictly positive
    return (np.nan, np.nan) if parameters[0] <= 0 else (1 / parameters[0], b)


def ours(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Estimate a and b using our cumulant approach.

    Parameters
    ----------
    x : np.ndarray
        The ground truth noise-free image
    y : np.ndarray
        The noisy image

    Returns
    -------
    a : float or nan
        Estimation of a
    b : float or nan
        Estimation of b
    """
    # Compute k statistics
    k2 = kstat(y, n=2)
    k3 = kstat(y, n=3)

    # Compute the values we need
    x_mean = x.mean()
    x_mean_squared = x.mean() ** 2
    x_squared_mean = np.power(x, 2).mean()
    x_mean_cubed = x.mean() ** 3
    x_cubed_mean = np.power(x, 3).mean()

    # Compute the coefficients of the polynomial
    coeffs = [
        x_cubed_mean - 3 * x_squared_mean * x_mean + 2 * x_mean_cubed - k3,
        3 * x_squared_mean - 3 * x_mean_squared,
        x_mean,
    ]

    # Create a polynomial and find its roots
    kz3 = np.polynomial.Polynomial(coeffs)
    roots = kz3.roots()

    a = 1 / (roots[np.where(roots > 0)])
    if a.size == 0:
        # a must be strictly positive
        return np.nan, np.nan
    else:
        a = a[0]

    # Compute b squared
    b_squared = k2 - x_squared_mean + x_mean_squared - x_mean / a

    # b squared can't be negative
    return (
        a,
        np.sqrt(b_squared) if b_squared >= 0 else 0,
    )


def log_likelihood(x, y, a, b, k_max=100) -> float:
    """
    Computes the log-likelihood for parameter values a and b. This is not
    trimmed for performance but we included this version for its simplicity.

    Parameters
    ----------
    a : float > 0
        estimation of the quantum efficiency
    b : float > 0
        estimation of the gaussian noise standard deviation
    k_max : int > 0
        The maximum value of k that we want to compute in the sum

    Returns
    -------
    log_likelihood : float
        $\mathcal{LL}(y|a,b,x) = \sum_{n}\log{\left(\sum_{k=0}^{k_max} \frac{(ax_n)^k}{k!b\sqrt{2\pi}}\exp{\left(-ax_n-\frac{(y_n-k/a)^2}{2b^2}\right)}\right)}$

    """
    # Prepare the computation
    n = x.shape[0]
    ks = np.tile(np.arange(k_max + 1), (n, 1))
    ys = np.tile(y, (k_max + 1, 1)).T
    mus = np.tile(a * x, (k_max + 1, 1)).T

    # Compute the the gaussian and Poisson parts
    gaussian_part = norm.pdf(ys, loc=ks / a, scale=b)
    poisson_part = poisson.pmf(ks, mu=mus)

    # Compute the likelihood for each pixel
    total = np.multiply(gaussian_part, poisson_part)
    per_pixel_likelihood = np.sum(total, axis=1)

    # If we have one pixel with zero likelihood, the total likelihood should be
    # zero too, hence the log one should be -infinity
    return (
        -np.inf
        if (per_pixel_likelihood <= 0).any()
        else np.sum(np.log(per_pixel_likelihood))
    )
