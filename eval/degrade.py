import numpy as np
from skimage.util import random_noise


def to01(img):
    img = img.astype(np.float64)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def add_gaussian(img01, var=0.01, seed=0):
    return random_noise(img01, mode="gaussian", var=var, rng=seed)


def add_salt_pepper(img01, amount=0.05, seed=0):
    return random_noise(img01, mode="s&p", amount=amount, rng=seed)


def add_speckle(img01, var=0.01, seed=0):
    return random_noise(img01, mode="speckle", var=var, rng=seed)


def low_light(img01, gamma=2.5, scale=0.35):
    """Simulate low-light capture: gamma-darken + reduce dynamic range,
    matching the LoL-style degradation Paper 3 works with."""
    darkened = (img01 ** gamma) * scale
    return np.clip(darkened, 0, 1)


def low_light_plus_noise(img01, gamma=2.5, scale=0.35, noise_var=0.004, seed=0):
    """The combined degradation neither Paper 2 nor Paper 3 tests:
    realistic smartphone/dermoscopic low-light capture with sensor noise."""
    dark = low_light(img01, gamma=gamma, scale=scale)
    noisy_dark = random_noise(dark, mode="gaussian", var=noise_var, rng=seed)
    return np.clip(noisy_dark, 0, 1)
