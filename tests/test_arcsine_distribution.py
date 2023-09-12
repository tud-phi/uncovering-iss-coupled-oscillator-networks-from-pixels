import matplotlib.pyplot as plt
import numpy as np


def test_arcsine_distribution(num_samples: int = 10000):
    rng = np.random.default_rng(seed=0)
    x = rng.uniform(0, 1, size=(num_samples,))
    y = np.sin(np.pi * x / 2)**2
    plt.hist(y, bins=100)
    plt.show()


if __name__ == "__main__":
    test_arcsine_distribution()
