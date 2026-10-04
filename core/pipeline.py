"""
Proposed method: noise- and illumination-adaptive fractal-fractional
edge detector with DFRWT subband-reliability weighting.

Pipeline:
  1. DFRWT-decompose the (possibly noisy, possibly low-light) image at a
     base fractional order -> {LL, LH, HL, HH}.
  2. Estimate a local reliability map per subband (MAD-based local SNR
     proxy) -> drives a per-pixel fractional order alpha (noise-adaptive).
  3. Estimate a local illumination map from LL -> drives a per-pixel
     fractal dimension beta (illumination-adaptive, extends Paper 3's
     purely gradient-driven beta).
  4. Apply the fractal-fractional Sobel (Paper 3 masks) OR the isotropic
     Riesz operator, with per-pixel (alpha, beta) / per-pixel Riesz order,
     within each subband.
  5. Fuse the subband edge maps, weighted by their reliability, back to
     full resolution; Otsu-threshold for a binary edge map.

Two operator backends are offered: 'df3sop' (fractal-fractional Sobel,
Paper 3 masks) and 'riesz' (isotropic Riesz derivative, this project's
extension).
"""
import numpy as np
from scipy.ndimage import uniform_filter, zoom
from skimage.filters import threshold_otsu

from core.dfrwt import dfrwt2, subband_magnitudes
from core.reliability import subband_reliability, reliability_to_order
from core.df3sop import adaptive_beta, df3sop_gradient_pixelwise_beta
from core.riesz import riesz_magnitude


def _resize_to(arr, shape):
    zy = shape[0] / arr.shape[0]
    zx = shape[1] / arr.shape[1]
    return zoom(arr, (zy, zx), order=1)


def illumination_adaptive_beta(subband_img: np.ndarray, win: int = 9):
    """Extends Paper 3's gradient-driven beta (Eq. 36) with an explicit
    local-illumination term: darker local regions get pushed toward more
    sensitive (lower) beta to compensate for weak raw gradients."""
    gx = np.gradient(subband_img, axis=1)
    gy = np.gradient(subband_img, axis=0)
    G_ref = np.sqrt(gx**2 + gy**2)
    beta_structural = adaptive_beta(G_ref)

    local_illum = uniform_filter(subband_img, size=win)
    lo, hi = local_illum.min(), local_illum.max()
    illum_norm = (local_illum - lo) / (hi - lo + 1e-8)

    beta = beta_structural * (0.5 + 0.5 * illum_norm)
    return np.clip(beta, 0.25, 0.95)


def proposed_edge_map(img01: np.ndarray, base_p: float = 0.5,
                       backend: str = "df3sop", win: int = 5,
                       edge_percentile: float = 10.0):
    """Full proposed pipeline. Returns (soft_edge_map, binary_edge_map)."""
    H, W = img01.shape
    subbands = dfrwt2(img01, base_p)
    mags = subband_magnitudes(subbands)
    reliab = subband_reliability(mags, win=win)

    fused = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    for name, sb_mag in mags.items():
        rel_map = reliab[name]
        alpha_map = reliability_to_order(rel_map, p_min=0.3, p_max=1.0)
        alpha_mean_levels = (0.3, 0.5, 0.7, 1.0)

        if backend == "df3sop":
            beta_map = illumination_adaptive_beta(sb_mag, win=win)
            # quantize alpha into a few levels, run df3sop per level, select
            edge_sb = np.zeros_like(sb_mag)
            idx = np.argmin(
                np.abs(alpha_map[..., None] - np.array(alpha_mean_levels)[None, None, :]),
                axis=-1,
            )
            for i, a in enumerate(alpha_mean_levels):
                m = idx == i
                if not np.any(m):
                    continue
                G_level = df3sop_gradient_pixelwise_beta(sb_mag, a, beta_map)
                edge_sb[m] = G_level[m]
        elif backend == "riesz":
            edge_sb = np.zeros_like(sb_mag)
            idx = np.argmin(
                np.abs(alpha_map[..., None] - np.array(alpha_mean_levels)[None, None, :]),
                axis=-1,
            )
            for i, a in enumerate(alpha_mean_levels):
                m = idx == i
                if not np.any(m):
                    continue
                # Riesz order mapped from fractional order range (0.3-1.0) to (0.5-1.5)
                riesz_order = 0.5 + a
                G_level = riesz_magnitude(sb_mag, riesz_order)
                edge_sb[m] = G_level[m]
        else:
            raise ValueError(backend)

        # Normalize each subband's edge response to [0,1] BEFORE fusion.
        # Without this, subbands whose quantized beta happened to land near
        # the low-beta (high-amplification) bucket dominate the weighted
        # sum purely on amplitude, not genuine edge content.
        edge_sb = edge_sb - edge_sb.min()
        denom = edge_sb.max()
        if denom > 1e-8:
            edge_sb = edge_sb / denom

        edge_sb_full = _resize_to(edge_sb, (H, W))
        rel_full = _resize_to(rel_map, (H, W))
        fused += edge_sb_full * (0.3 + 0.7 * rel_full)
        weight_sum += (0.3 + 0.7 * rel_full)

    soft = fused / (weight_sum + 1e-8)
    soft = (soft - soft.min()) / (soft.max() - soft.min() + 1e-8)

    # NOTE: Otsu assumes a roughly bimodal histogram (edge vs. background).
    # Multi-subband reliability-weighted fusion after bilinear upsampling of
    # coarse subbands does not preserve that bimodality -- it smears edge
    # energy into non-edge regions, so Otsu systematically under-thresholds
    # (we verified this empirically: fused-map median ~0.40 rather than
    # concentrated near 0). A percentile threshold calibrated to a typical
    # edge-pixel fraction is a more robust choice for this fused
    # representation and is what we use here.
    t = np.percentile(soft, 100 - edge_percentile)
    binary = soft > t
    return soft, binary
