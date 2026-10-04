import numpy as np
from skimage.feature import canny
from skimage.filters import threshold_otsu, sobel

from core.gl_sobel import gl_sobel_gradient
from core.df3sop import df3sop_gradient, adaptive_beta, df3sop_gradient_pixelwise_beta
from core.riesz import riesz_magnitude


def _otsu_binary(soft):
    soft = (soft - soft.min()) / (soft.max() - soft.min() + 1e-8)
    try:
        t = threshold_otsu(soft)
    except ValueError:
        t = 0.5
    return soft, soft > t


def method_canny(img01):
    binary = canny(img01, sigma=1.0)
    return binary.astype(float), binary


def method_sobel(img01):
    soft = sobel(img01)
    return _otsu_binary(soft)


def method_gl_fractional_sobel(img01, nu=0.8):
    G, _, _ = gl_sobel_gradient(img01, nu=nu)
    return _otsu_binary(G)


def method_df3sop_fixed(img01, alpha=1.0, beta=0.5):
    G, _, _ = df3sop_gradient(img01, alpha, beta)
    return _otsu_binary(G)


def method_df3sop_adaptive_beta(img01, alpha=1.0):
    gx = np.gradient(img01, axis=1)
    gy = np.gradient(img01, axis=0)
    G_ref = np.sqrt(gx**2 + gy**2)
    beta_map = adaptive_beta(G_ref)
    G = df3sop_gradient_pixelwise_beta(img01, alpha, beta_map)
    return _otsu_binary(G)


def method_riesz_fixed(img01, alpha=1.0):
    G = riesz_magnitude(img01, alpha)
    return _otsu_binary(G)
