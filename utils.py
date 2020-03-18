import numpy as np


def get_reg(regularization):
    if regularization.lower() == "l0":
        return lambda x, lam: lam * np.count_nonzero(x)
    elif regularization.lower() == "l1":
        return lambda x, lam: lam * np.sum(np.abs(x))
    # elif regularization.lower() == "cad":
    #     return prox_cad
    else:
        raise NotImplementedError("{} has not been implemented".format(regularization))


def add_noise(x, std=1e-3):
    if isinstance(x, list):
        x_noisy = []
        for xi in x:
            x_noisy.append(xi + std*np.random.standard_normal(xi.shape))
        return x_noisy
    else:
        return x + std*np.random.standard_normal(x.shape)


def corrupt_data(x, corrupt_fraction=.01, corruption_strength=None):
    if corruption_strength is None:
        corruption_strength = 5*np.std(x)

    if isinstance(x, list):
        corrupted_idxs = []
        for xi in x:
            n_corrupted_samples = int(corrupt_fraction*xi.shape[0])
            corrupted_samples = np.random.choice(xi.shape[0]-2, size=n_corrupted_samples, replace=False) + 1
            xi[corrupted_samples] += corruption_strength*(np.random.rand(n_corrupted_samples, xi.shape[1])-.5)
            corrupted_idxs.append(corrupted_samples)
    else:
        n_corrupted_samples = int(corrupt_fraction*x.shape[0])
        corrupted_idxs = np.random.choice(x.shape[0]-2, size=n_corrupted_samples, replace=False) + 1
        x[corrupted_idxs] += corruption_strength*(np.random.rand(n_corrupted_samples, x.shape[1])-.5)
    return corrupted_idxs


def sigma_plot_sparsity(sigma):
    sigma_inf = sigma.copy()
    sigma_inf[sigma_inf==0] = np.inf
    return sigma_inf
