import numpy as np
from scipy.stats import kstat


class PGNoise:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        x : array, size n
            array of the real values of the image
        y : array, size n
            array of the values with the poisson and gaussian noise
        """
        self.__x = x
        self.__y = y
        self.__n = len(np.atleast_1d(y))

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def __str__(self):
        return f"n : {self.__n}"

    def variance_method(self):
        """
        Estimate (a,b) based on the variance of different pixels
        In this method, we suppose that for each unique value of x_i, the average of y_i is indeed x_i. This method is biased
        Parameters
        ----------

        Returns
        a : float or nan
            Estimation of a
        b : float or nan
            Estimation of b
        """
        argsort = np.argsort(self.__x)
        x_sorted = self.__x[argsort]
        y_sorted = self.__y[argsort]
        x_unique, x_index, xc = np.unique(
            x_sorted, return_index=True, return_counts=True
        )
        splitted = np.split(y_sorted - x_sorted, x_index[1:])
        var = np.zeros(x_unique.size)
        for i, element in enumerate(splitted):
            var[i] = np.square(element).mean()  # Because mean = 0
        lhs = np.array([x_unique, np.ones(x_unique.size)]).T
        lhs_w = lhs * np.sqrt(xc[:, np.newaxis])
        rhs_w = var * np.sqrt(xc)
        parameters, residuals, rank, s = np.linalg.lstsq(lhs_w, rhs_w, rcond=None)
        if parameters[1] < 0:
            lhs_w = np.array(lhs_w[:, 0])[:, None]
            parameters, residuals, rank, s = np.linalg.lstsq(lhs_w, rhs_w, rcond=None)
            a = 1 / parameters[0]
            b = 0
        else:
            a = 1 / parameters[0]
            b = np.sqrt(parameters[1])
        return a, b

    def cumulant_method(self):
        """
        See derivation, it turns out this is an okay way to measure a and b but we don't use the fact that each pixel of y is dependant form a pixel of x
        Parameters
        ----------

        Returns
        a : float or nan
            Estimation of a
        b : float or nan
            Estimation of b
        """
        k2 = kstat(self.__y, n=2)
        k3 = kstat(self.__y, n=3)

        x_mean = self.__x.mean()
        x_mean_squared = self.__x.mean() ** 2
        x_squared_mean = np.power(self.__x, 2).mean()
        x_mean_cubed = self.__x.mean() ** 3
        x_cubed_mean = np.power(self.__x, 3).mean()

        coeffs = [
            x_cubed_mean - 3 * x_squared_mean * x_mean + 2 * x_mean_cubed - k3,
            3 * x_squared_mean - 3 * x_mean_squared,
            x_mean,
        ]
        kz3 = np.polynomial.Polynomial(coeffs)
        roots = kz3.roots()
        # Roots contain values for a^-1

        a = 1 / (roots[np.where(roots > 0)])
        if a.size == 0:
            return np.nan, np.nan
        else:
            a = a[0]
        b2 = k2 - x_squared_mean + x_mean_squared - x_mean / a
        return a, np.sqrt(b2) if b2 >= 0 else 0
