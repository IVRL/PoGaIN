# PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

Authors: Nicolas Bähler, Majed El Helou, Étienne Objois, Kaan Okumuş, and Sabine
Süsstrunk, _Fellow_, _IEEE_.

## [[IEEE Xplore](https://ieeexplore.ieee.org/document/9976220)] - [[Paper PDF](https://github.com/IVRL/PoGaIN/blob/main/paper/paper.pdf)] - [[Abridged Supplementary Material PDF](https://github.com/IVRL/PoGaIN/blob/main/supplementary_material/abriged_supplementary_material.pdf)] - [[Complete Supplementary Material PDF](https://github.com/IVRL/PoGaIN/blob/main/supplementary_material/supplementary_material.pdf)]

## Abstract

Image noise can often be accurately fitted to a Poisson-Gaussian distribution. However, estimating the distribution parameters from a noisy image only is a challenging task. Here, we study the case when paired noisy and noise-free samples are accessible. No method is currently available to exploit the noise-free information, which may help to achieve more accurate estimations. To fill this gap, we derive a novel, cumulant-based, approach for Poisson-Gaussian noise modeling from paired image samples. We show its improved performance over different baselines, with special emphasis on MSE, effect of outliers, image dependence, and bias. We additionally derive the log-likelihood function for further insights and discuss real-world applicability.

## Requirements

For this code base we used Python 3.9. More detailed package requirements can be
found in the [`environment.yml`](https://github.com/IVRL/PoGaIN/blob/main/environment.yml) file which can directly be used to build an
anaconda environment.

## Introduction

For this paper, we use a noise model based on the Poisson-Gaussian noise model introduced by Foi _et
al._ \[1\] to model noise that arises in imaging processes. Let us denote the
observed noisy image as $y$ and the ground-truth noise-free image as $x$. Then,
the Poisson-Gaussian model takes the following form:

$$
\begin{equation}
    y = \frac{1}{a} \alpha + \beta, \quad \alpha \sim \mathcal{P}(ax), \quad \beta \sim \mathcal{N}(0,b^2).
\end{equation}
$$

For example, one might have an instance like:

![image info](images/comparison.png)

where $a = 11$ and $b = 0.01$.

Our method (_Ours_) then estimates those parameters based on the noisy and noise-free
image pair using the cumulant expansion. As a baseline we implement another estimator based on variance
(_Var_) which also uses such image pairs. For the above example, we get the
estimates and the log-likelihood below:

```shell
===============
Ground truth:
a=11
b=0.01
===============
Log-likelihood:
LL=153825.966
===============
Var:
a=11.08914
b=0.02098
===============
Ours:
a=10.97338
b=0.00735
===============
```

You can find the code for this example in the [`code/example.py`](https://github.com/IVRL/PoGaIN/blob/main/code/example.py) or
[`code/example.ipynb`](https://github.com/IVRL/PoGaIN/blob/main/code/example.ipynb) file. The implementation of the methods and the
log-likelihood on the other hand can be found in [`code/implementations.py`](https://github.com/IVRL/PoGaIN/blob/main/code/implementations.py).

## Reproducibility

We provide the code that we used to compute our results, figures and tables which can be found in the
[`code/reproducibility/`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty) directory.

Here, we want to give a quick overview of that code section and comment on some
aspects that might not be self explanatory.

```shell
reproducibility
├── cnn
│   ├── model
│   └── model.py
├── compute_figures_and_tables.py
├── compute_results.py
├── csvs
│  (├── cnn.csv)
│  (├── foi.csv)
│  (├── ours.csv)
│  (├── real.csv)
│   ├── results.csv
│  (└── var.csv)
├── figures
│   ├── bias.png
│   ├── k_plot.png
│   ├── ll.png
│   ├── mse_dependence.png
│   └── mse.png
├── our_results
│   ├── model
│   └── results.csv
├── tables
│   ├── error_a.tex
│   ├── error_b.tex
│   ├── outliers_a.tex
│   ├── outliers_b.tex
│   └── outliers_combined.tex
└── w2s
    ├── code
    ├── data
    ├── data_bridge.py
    ├── documentation
    ├── get_preds.m
    ├── README.md
    └── results
```

### General comments

The two top level python scripts [`compute_results.py`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/compute_results.py) and
[`compute_figures_and_tables.py`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/compute_figures_and_tables.py) are computing the results as csv files,
outputting figures as png images and tables as tex files in the respective
directories. For the paper, we merged the tables on outliers into one tex file.

Note that we advice to split the computation of the results into different runs
because it is quite time consuming. For that purpose, we provide a set of
functions where each computes the results for a specific method, saved in a
separate csv file in [`csvs`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/csvs). Further, the
log-likelihood computation is also split into different functions for each
method. Finally, we provide a function that combines all those resulting csv
files into one.

If you don't want to compute the results yourself, you can also use the results
we computed on a cluster and provide in the
[`our_results`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/our_results)
directory. The current csv file
[`csvs/results.csv`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/csvs/results.csv)
is a copy of that file

### Important remarks

The _FOI_ method **requires Matlab to be installed**! We decided to implement an easy way of
using the Matlab code \[2\] in [`w2s`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/w2s) which we took from
[`GitHub`](https://github.com/IVRL/w2s). You do not need to get into the code
itself. Running the [`compute_results.py`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/compute_results.py) will do everything for you.

The _CNN_ method involves **training the neural network** as well. Again, we provide
a method for this inside the [`compute_results.py`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/compute_results.py) script. Alternatively, you
can also use the pretrained model that we provide in the [`our_results`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/our_results)
directory. In order to do so simply copy the [`our_results/model`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/our_results/model) directory into the
[`cnn`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/cnn) directory. It will then be loaded by the method which is called in
[`compute_results.py`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/compute_results.py)
to compute the results for _CNN_.

The
[`k_plot.png`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/figures/k_plots.png)
figure **is not computed again by default** because it also
requires some time. You can easily change this in
[`compute_figures_and_tables.py`](https://github.com/IVRL/PoGaIN/blob/main/code/reproducibilty/compute_figures_and_tables.py).

## Citation

```bibtex
@article{9976220,
  author  = {B{\"a}hler, Nicolas and El Helou, Majed and Objois, {\'E}tienne and Okumu{\c{s}}, Kaan and S{\"u}sstrunk, Sabine},
  doi     = {10.1109/LSP.2022.3227522},
  journal = {IEEE Signal Processing Letters},
  number  = {},
  pages   = {2602-2606},
  title   = {PoGaIN: Poisson-Gaussian Image Noise Modeling From Paired Samples},
  volume  = {29},
  year    = {2022}
}
```

## References

\[1\] [https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf)

\[2\] [https://arxiv.org/pdf/2003.05961.pdf](https://arxiv.org/pdf/2003.05961.pdf)
