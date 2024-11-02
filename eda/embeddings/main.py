"""
Take in a list of embedding vectors and output statistics on them.

The embedding file must be given as the first argument.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy


def sphere_marginal(d: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the marginal distribution for a uniform distribution on the sphere
    at the points in x.

    See: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    """
    k = (d - 1) / 2
    return (1 - x**2) ** (k - 1) / scipy.special.beta(0.5, k)


def main(vecs: np.ndarray) -> None:

    T = vecs.shape[0]
    D = vecs.shape[1]

    mean = np.mean(vecs, axis=0)
    rayleigh_statistic = T * D * np.dot(mean, mean)
    rayleigh_pvalue = 1 - scipy.stats.chi2.cdf(rayleigh_statistic, D)
    print(f"Mean length:        {np.linalg.norm(mean):.2f}")
    print(f"Rayleigh statistic: {rayleigh_statistic}")
    print(f"Rayleigh p-value:   {rayleigh_pvalue:.4f}")

    dot_test_vecs = vecs - mean
    np.random.default_rng().shuffle(dot_test_vecs)
    dot_test_vecs = dot_test_vecs[:2000]
    dot_test = np.einsum("id,jd->ij", dot_test_vecs, dot_test_vecs).reshape((-1,))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(dot_test, bins=100, density=True)
    ax.plot(np.linspace(-1, 1, 1000), sphere_marginal(D, np.linspace(-1, 1, 1000)))
    ax.set_title("Dot Product Distribution")
    ax.set_xlim(-1, 1)
    fig.savefig("embed-dot-test.png")

    pca_matrix = np.einsum("ti,tj->ij", vecs - mean, vecs - mean) / T
    pca_eigvals, _ = np.linalg.eigh(pca_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pca_eigvals[::-1])
    ax.set_title("PCA Eigenvalues")
    ax.set_yscale("log")
    fig.savefig("embed-eigenvalues.png")


if __name__ == "__main__":
    main(np.load(sys.argv[1]))
