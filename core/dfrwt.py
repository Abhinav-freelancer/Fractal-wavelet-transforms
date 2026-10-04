"""
2-D Discrete Fractional Wavelet Transform (DFRWT), single level.

Implements the chirp-modulation / DWT / chirp-demodulation sandwich
described in Yang & Li (ESWA 2026), Eqs. (10)-(18):

  1. Modulate the (real) image with a 2-D chirp depending on fractional
     order p (angle alpha = p*pi/2).
  2. Run a standard 2-D DWT (Haar, via PyWavelets) on the complex,
     modulated signal -> four complex subbands LL, LH, HL, HH.
  3. Demodulate each subband with the matching chirp at the new (coarser)
     index scale.

The magnitude of the resulting complex subbands is what gets fed to the
edge-detection stage, matching the paper's visual outputs (Fig. 3/5) which
display magnitude images.

Note: this is a faithful *practical* implementation of the method as
described; it is not claimed to be bit-exact with any reference codebase,
since the paper gives no code. It reproduces the qualitative behaviour
reported in the paper (order p controls subband sparsity / energy
redistribution) which is what we validate against in the sanity check.
"""
import numpy as np
import pywt


def _chirp_2d(shape, p, j=0):
    """2-D chirp modulation array for scale level j, order p."""
    alpha = p * np.pi / 2.0
    cot_a = 1.0 / np.tan(alpha) if np.sin(alpha) != 0 else 0.0
    H, W = shape
    n1 = np.arange(H).reshape(-1, 1)
    n2 = np.arange(W).reshape(1, -1)
    scale = 2.0 ** j
    phase = (1.0 / 8.0) * (((n1 * scale) ** 2 + (n2 * scale) ** 2)) * cot_a
    return np.exp(1j * phase)


def _chirp_2d_out(shape, p, j=1):
    """Demodulation chirp at the decomposed (coarser) resolution."""
    alpha = p * np.pi / 2.0
    cot_a = 1.0 / np.tan(alpha) if np.sin(alpha) != 0 else 0.0
    H, W = shape
    k1 = np.arange(H).reshape(-1, 1)
    k2 = np.arange(W).reshape(1, -1)
    scale = 2.0 ** j
    phase = 0.5 * (((k1 * scale) ** 2 + (k2 * scale) ** 2)) * cot_a
    return np.exp(-1j * phase)


def dfrwt2(img: np.ndarray, p: float, wavelet: str = "haar"):
    """Single-level 2-D DFRWT.

    Returns dict with complex subbands {LL, LH, HL, HH}.
    p in (0, 1]; p=1 reduces to standard DWT (alpha = pi/2, cot=0 -> no
    modulation phase), matching the paper's statement that DFRWT == DWT
    when p = 1.
    """
    img = img.astype(np.float64)
    chirp_in = _chirp_2d(img.shape, p, j=0)
    modulated = img * chirp_in

    # complex 2D DWT = separate real/imag DWT
    re = modulated.real
    im = modulated.imag
    cA_r, (cH_r, cV_r, cD_r) = pywt.dwt2(re, wavelet)
    cA_i, (cH_i, cV_i, cD_i) = pywt.dwt2(im, wavelet)

    LL = cA_r + 1j * cA_i
    LH = cH_r + 1j * cH_i
    HL = cV_r + 1j * cV_i
    HH = cD_r + 1j * cD_i

    chirp_out = _chirp_2d_out(LL.shape, p, j=1)
    LL = LL * chirp_out
    LH = LH * chirp_out
    HL = HL * chirp_out
    HH = HH * chirp_out

    return {"LL": LL, "LH": LH, "HL": HL, "HH": HH}


def idfrwt2(subbands: dict, p: float, out_shape, wavelet: str = "haar"):
    """Inverse single-level 2-D DFRWT (approximate reconstruction)."""
    chirp_out = _chirp_2d_out(subbands["LL"].shape, p, j=1)
    # undo the output chirp (its magnitude is 1, so conjugate inverts it)
    LL = subbands["LL"] * np.conj(chirp_out)
    LH = subbands["LH"] * np.conj(chirp_out)
    HL = subbands["HL"] * np.conj(chirp_out)
    HH = subbands["HH"] * np.conj(chirp_out)

    re = pywt.idwt2((LL.real, (LH.real, HL.real, HH.real)), wavelet)
    im = pywt.idwt2((LL.imag, (LH.imag, HL.imag, HH.imag)), wavelet)
    recon_mod = re + 1j * im
    recon_mod = recon_mod[: out_shape[0], : out_shape[1]]

    chirp_in = _chirp_2d(out_shape, p, j=0)
    recon = recon_mod * np.conj(chirp_in)
    return recon.real


def subband_magnitudes(subbands: dict):
    return {k: np.abs(v) for k, v in subbands.items()}
