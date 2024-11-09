"""
Take in a list of embedding vectors and output statistics on them.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml


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

    DOT_TEST_NUM = 2000
    dot_test_vecs = vecs - mean
    np.random.default_rng().shuffle(dot_test_vecs)
    dot_test_vecs = dot_test_vecs[:DOT_TEST_NUM]
    dot_test = np.einsum("id,jd->ij", dot_test_vecs, dot_test_vecs).reshape((-1,))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(dot_test, bins=100, density=True)
    ax.plot(np.linspace(-1, 1, 1000), sphere_marginal(D, np.linspace(-1, 1, 1000)))
    ax.set_title(f"Dot Product Distribution for {D}D")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(-1, 1)
    fig.savefig("embed-dot-test.png")

    pca_matrix = np.einsum("ti,tj->ij", vecs - mean, vecs - mean) / T
    pca_eigvals, _ = np.linalg.eigh(pca_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pca_eigvals[::-1])
    ax.set_title(f"Principal Component Analysis for {D}D")
    ax.set_yscale("log")
    ax.set_ylabel("Eigenvalue")
    fig.savefig("embed-eigenvalues.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze embeddings.")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        help="Dimension of the embedding vectors",
        default=256,
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    plt.style.use(config["plotting"]["style"])

    main(np.load(config["data"]["embeddings"][args.dimension]))
