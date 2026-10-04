"""
Grunwald-Letnikov fractional-order Sobel operator.
Reproduces Eq. (19)-(25) and the four-directional masks from:
Yang & Li, "A new edge detection method for noisy image based on discrete
fractional wavelet transform and improved Canny algorithm", ESWA 2026.

The four 3x3 masks S^nu_0, S^nu_45, S^nu_90, S^nu_135 are given explicitly
in the paper (Section 3.1). We hard-code them from the printed formulas.
"""
import numpy as np
from scipy.ndimage import convolve


def gl_sobel_masks(nu: float):
    """Return the four fractional-order Sobel masks (0,45,90,135 deg).

    nu is the fractional order (paper uses nu = 0.8). At nu = 1 these
    should reduce to (a scaled version of) the classical isotropic Sobel.
    """
    s2 = np.sqrt(2.0)

    S0 = 2.0 * np.array([
        [(nu - nu**2) / 2.0,           nu,  -1.0],
        [s2 * (nu - nu**2) / 2.0,   s2*nu,  -s2],
        [(nu - nu**2) / 2.0,           nu,  -1.0],
    ])

    S45 = np.array([
        [s2*(nu - nu**2),        2 + 2*s2*nu,              0.0],
        [nu - nu**2,                     0.0,     -(2 + 2*s2*nu)],
        [0.0,                      nu**2 - nu,     s2*(nu**2 - nu)],
    ])

    S90 = 2.0 * np.array([
        [-1.0,                       -s2,                -1.0],
        [nu,                       s2*nu,                  nu],
        [(nu - nu**2) / 2.0,  s2*(nu - nu**2) / 2.0, (nu - nu**2) / 2.0],
    ])

    S135 = np.array([
        [0.0,                 2 + 2*s2*nu,      s2*(nu - nu**2)],
        [-(2 + 2*s2*nu),              0.0,          nu - nu**2],
        [s2*(nu**2 - nu),      nu**2 - nu,                  0.0],
    ])

    return {"0": S0, "45": S45, "90": S90, "135": S135}


def gl_sobel_gradient(img: np.ndarray, nu: float = 0.8):
    """Compute gradient magnitude and the four directional responses."""
    masks = gl_sobel_masks(nu)
    responses = {}
    for name, k in masks.items():
        responses[name] = convolve(img.astype(np.float64), k, mode="reflect")
    G = np.sqrt(sum(r**2 for r in responses.values()))
    phi = np.arctan2(responses["135"], responses["0"])
    return G, phi, responses


def classical_sobel_isotropic():
    """Isotropic Sobel (coefficient sqrt(2) instead of 2) for nu=1 sanity check."""
    return {
        "0": np.array([[1, 0, -1], [np.sqrt(2), 0, -np.sqrt(2)], [1, 0, -1]]),
    }
