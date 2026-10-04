"""
Subband reliability analysis for DFRWT subbands.

Standard wavelet-domain noise estimation uses the robust MAD (median
absolute deviation) estimator on the finest detail subband:

    sigma_hat = median(|HH|) / 0.6745

This is a per-subband scalar noise estimate. We extend it to a *local*
reliability map by computing a sliding-window local MAD, and define a
reliability score in [0, 1] as an inverse function of estimated local
noise relative to local signal energy (a local SNR proxy).

This reliability map is the signal used to drive the per-subband
fractional order (and, downstream, illumination-driven beta) in the
adaptive fusion pipeline.
"""
import numpy as np
from scipy.ndimage import uniform_filter, median_filter


def mad_noise_sigma(detail_subband: np.ndarray) -> float:
    """Global robust noise sigma estimate from a detail (HH) subband."""
    return np.median(np.abs(detail_subband)) / 0.6745


def local_reliability_map(subband_mag: np.ndarray, win: int = 5) -> np.ndarray:
    """Local SNR-proxy reliability in [0,1] for a subband magnitude image.

    local_signal = local mean magnitude
    local_noise  = local MAD-based sigma (via local median filter residual)
    reliability  = local_signal / (local_signal + local_noise + eps)
    """
    local_mean = uniform_filter(subband_mag, size=win)
    local_med = median_filter(subband_mag, size=win)
    resid = np.abs(subband_mag - local_med)
    local_noise = median_filter(resid, size=win) / 0.6745
    eps = 1e-8
    reliability = local_mean / (local_mean + local_noise + eps)
    # normalize to [0,1]
    rmin, rmax = reliability.min(), reliability.max()
    if rmax - rmin > 1e-8:
        reliability = (reliability - rmin) / (rmax - rmin)
    return reliability


def subband_reliability(subbands_mag: dict, win: int = 5) -> dict:
    """Reliability map per subband {LL, LH, HL, HH}."""
    return {k: local_reliability_map(v, win=win) for k, v in subbands_mag.items()}


def reliability_to_order(reliability_map: np.ndarray, p_min: float = 0.2, p_max: float = 1.0) -> np.ndarray:
    """Map local reliability (noisier -> lower reliability) to a local
    fractional order p. Noisy/unreliable regions get a *lower* p (paper 2's
    Table 2 / Fig.10 shows low-noise, low-p favors stronger noise
    suppression in the DFRWT framework); clean/reliable regions get p
    closer to 1 (closer to classical behaviour, sharper edges).
    """
    return p_min + (p_max - p_min) * reliability_map
