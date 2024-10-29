"""
Take in a list of embedding vectors and output statistics on them.

The embedding file must be given as the first argument.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main(vecs: np.ndarray) -> None:

    T = vecs.shape[0]
    D = vecs.shape[1]

    mean = np.mean(vecs, axis=0)
    rayleigh_statistic = T * D * np.dot(mean, mean)
    rayleigh_pvalue = 1 - stats.chi2.cdf(rayleigh_statistic, D)
    print(f"Mean length:        {np.linalg.norm(mean):.2f}")
    print(f"Rayleigh statistic: {rayleigh_statistic}")
    print(f"Rayleigh p-value:   {rayleigh_pvalue:.4f}")

    pca_matrix = np.einsum("ti,tj->ij", vecs - mean, vecs - mean) / T
    pca_eigvals, _ = np.linalg.eigh(pca_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pca_eigvals[::-1])
    ax.set_yscale("log")
    fig.savefig("embed-eigenvalues.png")


if __name__ == "__main__":
    main(np.load(sys.argv[1]))
