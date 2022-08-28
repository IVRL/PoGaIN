# PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

Authors: Nicolas Bähler, Étienne Objois, Kaan Okumuş, [Majed El Helou](https://majedelhelou.github.io/), and Sabine
Süsstrunk.

## [[Paper](https://www.google.ch/)] - [[Abridged Supplementary Material](https://www.google.ch/)] - [[Complete Supplementary Material](https://www.google.ch/)]

## Abstract

Image noise modeling is important for analyzing datasets and quantifying the
performance of image acquisition setups. It is consequently important for the
fundamental image denoising task. In practice, image noise can often be
accurately fitted to a Poisson-Gaussian distribution, whose parameters need to
be estimated.

Estimating the distribution parameters from a noisy image is a challenging task
studied in the literature. However, when paired noisy and noise-free samples are
available, no method is available to exploit this noise-free information to
obtain more accurate estimates. We derive variance- and cumulant-based
approaches for Poisson-Gaussian noise modeling from paired image samples. We
analyze our method in depth, show its improved performance over different
baselines, and additionally derive the log-likelihood function for further
insight.

## Requirements

For this code base we used Python 3.9. More detailed package requirements can be
found in the [`environment.yml`](https://github.com/IVRL/PoGaIN/blob/main/environment.yml) file which can directly be used to build an
anaconda environment.

## Introduction

For this paper, we use a Poisson-Gaussian noise model introduced by Foi _et
al._ \[1\] to model noise that arises in an imaging process. Let us denote the
observed noisy image as $y$ and the ground-truth noise-free image as $x$. Then,
the Poisson-Gaussian model takes the form of the following equation:

$$
\begin{equation}
    y = \frac{1}{a} \alpha + \beta, \quad \alpha \sim \mathcal{P}(ax), \quad \beta \sim \mathcal{N}(0,b^2).
\end{equation}
$$

For example, one might have an instance like:

![image info](comparison.png)

where $a=45, b=0.015$.

Our method then estimates those parameters based on the noisy and noise-free
image pair. For the above example, the estimated parameters are:

```shell
Ground-truth:
a=45
b=0.015
===============
Log-likelihood:
LL=135530.152
===============
Ours_v:
a=44.96868
b=0.01517
===============
Ours_c:
a=45.3798
b=0.01819
===============
```

## Citation

Citation

## Acknowledgements

Acknowledgements

## References

\[1\] [https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf)

