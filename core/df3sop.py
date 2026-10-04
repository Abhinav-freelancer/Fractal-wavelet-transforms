"""
Dual-Functional Fractal-Fractional Sobel Operator (DF3SOp), from:
Gowtham et al., "Dual-functional fractal-fractional Sobel operator for
efficient image enhancement and edge detection", Pattern Recognition 2026.

Masks M^{alpha,beta}_x, M^{alpha,beta}_y reproduced from Table 1 of the
paper. alpha = fractional order (paper fixes alpha=1 for edge detection,
Remark 1), beta = fractal dimension in (0,1] (adaptive, per-pixel).
At alpha=beta=1 these reduce to the classical (non-isotropic-coefficient)
Sobel operator.
"""
import numpy as np
from scipy.ndimage import convolve


def df3sop_masks(alpha: float, beta: float):
    b_a = beta ** alpha
    Mx = np.array([
        [-alpha / (2 * b_a),       (alpha * (1 - alpha)) / (2 * b_a),  alpha / (2 * b_a)],
        [-alpha / b_a,             (alpha * (1 - alpha)) / b_a,        alpha / b_a],
        [-alpha / (2 * b_a),       (alpha * (1 - alpha)) / (2 * b_a),  alpha / (2 * b_a)],
    ])
    My = np.array([
        [1 / (2 * b_a),            1 / b_a,                            1 / (2 * b_a)],
        [(1 - alpha) / (2 * b_a),  (1 - alpha) / b_a,                  (1 - alpha) / (2 * b_a)],
        [-alpha / (2 * b_a),       -alpha / b_a,                       -alpha / (2 * b_a)],
    ])
    return Mx, My


def df3sop_gradient(img: np.ndarray, alpha: float, beta: float):
    Mx, My = df3sop_masks(alpha, beta)
    Gx = convolve(img.astype(np.float64), Mx, mode="reflect")
    Gy = convolve(img.astype(np.float64), My, mode="reflect")
    G = np.sqrt(Gx**2 + Gy**2)
    return G, Gx, Gy


def adaptive_beta(G_ref: np.ndarray):
    """Pixel-wise adaptive beta per Eq. (36) of the paper.

    G_ref: a reference gradient-magnitude image used to classify each
    pixel as low/moderate/high gradient.
    """
    gmin, gmax, gavg = G_ref.min(), G_ref.max(), G_ref.mean()
    F1 = (gmin + gavg) / 2.0
    F2 = (gmax + gavg) / 2.0

    beta = np.full_like(G_ref, 0.5, dtype=np.float64)
    moderate = (G_ref > F1) & (G_ref <= F2)
    high = G_ref > F2

    denom = (gmax - gmin) if (gmax - gmin) > 1e-8 else 1e-8
    beta[moderate] = 0.1 + 0.7 * ((G_ref[moderate] - gmin) / denom)
    beta[high] = 0.1
    return beta


def df3sop_gradient_pixelwise_beta(img: np.ndarray, alpha: float, beta_map: np.ndarray,
                                    beta_levels=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Approximate per-pixel-beta convolution by computing full-image
    responses at a small set of beta levels and selecting per pixel
    (quantized adaptive-beta, computationally tractable)."""
    H, W = img.shape
    best_G = np.zeros((H, W))
    # precompute responses at each candidate beta, then pick nearest level per pixel
    responses = {}
    for b in beta_levels:
        G, _, _ = df3sop_gradient(img, alpha, b)
        responses[b] = G
    levels = np.array(beta_levels)
    idx = np.argmin(np.abs(beta_map[..., None] - levels[None, None, :]), axis=-1)
    for i, b in enumerate(beta_levels):
        mask = idx == i
        best_G[mask] = responses[b][mask]
    return best_G
