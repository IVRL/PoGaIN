import numpy as np
from scipy.stats import poisson, norm, kstat

# TODO rename methods, from ours_v to PoGAIN_v etc, also in paper


def ours_v(x: np.ndarray, y: np.ndarray):
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
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    x_unique, x_index, xc = np.unique(x_sorted, return_index=True, return_counts=True)
    splitted = np.split(y_sorted - x_sorted, x_index[1:])

    var = np.zeros(x_unique.size)
    for i, element in enumerate(splitted):
        var[i] = np.square(element).mean()

    lhs = np.array([x_unique, np.ones(x_unique.size)]).T
    lhs_w = lhs * np.sqrt(xc[:, np.newaxis])
    rhs_w = var * np.sqrt(xc)

    parameters, _, _, _ = np.linalg.lstsq(lhs_w, rhs_w, rcond=None)

    if parameters[1] < 0:  # One can argue that we should return nan here
        # Assuming b==0, redo the estimation with b==0
        lhs_w = np.array(lhs_w[:, 0])[:, None]
        parameters, _, _, _ = np.linalg.lstsq(lhs_w, rhs_w, rcond=None)

        b = 0
    else:
        b = np.sqrt(parameters[1])

    if parameters[0] <= 0:
        return np.nan, np.nan

    a = 1 / parameters[0]

    return a, b


def ours_c(x: np.ndarray, y: np.ndarray):
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
    k2 = kstat(y, n=2)
    k3 = kstat(y, n=3)

    x_mean = x.mean()
    x_mean_squared = x.mean() ** 2
    x_squared_mean = np.power(x, 2).mean()
    x_mean_cubed = x.mean() ** 3
    x_cubed_mean = np.power(x, 3).mean()

    coeffs = [
        x_cubed_mean - 3 * x_squared_mean * x_mean + 2 * x_mean_cubed - k3,
        3 * x_squared_mean - 3 * x_mean_squared,
        x_mean,
    ]
    kz3 = np.polynomial.Polynomial(coeffs)
    roots = kz3.roots()

    a = 1 / (roots[np.where(roots > 0)])
    if a.size == 0:
        return np.nan, np.nan
    else:
        a = a[0]
    b_squared = k2 - x_squared_mean + x_mean_squared - x_mean / a

    return (
        a,
        np.sqrt(b_squared) if b_squared >= 0 else 0,
    )


def log_likelihood(x, y, a, b, k_max=100):
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

    n = x.shape[0]

    ks = np.tile(np.arange(k_max + 1), (n, 1))
    ys = np.tile(y, (k_max + 1, 1)).T
    mus = np.tile(a * x, (k_max + 1, 1)).T

    gaussian_pdf = norm.pdf(ys, loc=ks / a, scale=b)
    poisson_pmf = poisson.pmf(ks, mu=mus)

    total_pmf = np.multiply(gaussian_pdf, poisson_pmf)
    per_pixel = np.sum(total_pmf, axis=1)

    # If we have one pixel with zero likelihood, the total likelihood should be
    # zero too, hence the log one should be -infinity
    return -np.inf if (per_pixel <= 0).any() else np.sum(np.log(per_pixel))
