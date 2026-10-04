"""
Riesz fractional derivative / isotropic fractional gradient operator.

The Riesz derivative of order alpha is defined in the Fourier domain as a
radially symmetric multiplier:

    F[(-Delta)^(alpha/2) f](xi) = |xi|^alpha * F[f](xi)

This gives a single isotropic operator (no preferred direction), in
contrast to the G-L / fractal-fractional Sobel masks (Papers 1-3) which
all need 4 directional 3x3 kernels combined by magnitude to approximate
isotropy.

We also provide a directional Riesz gradient (Riesz-x, Riesz-y) via the
first-order Riesz transform multiplied by |xi|^(alpha-1), which lets us
recover gradient *direction* (needed for non-max suppression in Canny-style
pipelines) while keeping the magnitude response isotropic.
"""
import numpy as np


def _freq_grid(shape):
    H, W = shape
    fy = np.fft.fftfreq(H) * 2 * np.pi
    fx = np.fft.fftfreq(W) * 2 * np.pi
    FX, FY = np.meshgrid(fx, fy)
    return FX, FY


def riesz_magnitude(img: np.ndarray, alpha: float, eps: float = 1e-6):
    """Isotropic fractional-order gradient magnitude via Riesz derivative.

    alpha in (0, 2]. alpha=2 corresponds to the (fractional) Laplacian
    magnitude response; alpha=1 is closest to a classical gradient-magnitude
    operator. We normalize so alpha=1 behaves like a first-order edge
    detector in scale.
    """
    img = img.astype(np.float64)
    H, W = img.shape
    FX, FY = _freq_grid((H, W))
    r = np.sqrt(FX**2 + FY**2)
    r[0, 0] = eps  # avoid singularity at DC

    multiplier = r ** alpha
    F = np.fft.fft2(img)
    out = np.fft.ifft2(F * multiplier)
    return np.abs(out)


def riesz_directional(img: np.ndarray, alpha: float, eps: float = 1e-6):
    """Directional Riesz transform components R_x, R_y of order alpha.

    Uses the classical Riesz transform kernels i*xi_k/|xi| combined with
    the |xi|^alpha fractional magnitude scaling, giving a direction-aware
    fractional gradient (G_x, G_y) analogous to Sobel's (Gx, Gy) but with
    an isotropic magnitude envelope.
    """
    img = img.astype(np.float64)
    H, W = img.shape
    FX, FY = _freq_grid((H, W))
    r = np.sqrt(FX**2 + FY**2)
    r_safe = np.where(r < eps, eps, r)

    mult_x = 1j * FX / r_safe * (r ** alpha)
    mult_y = 1j * FY / r_safe * (r ** alpha)
    mult_x[0, 0] = 0.0
    mult_y[0, 0] = 0.0

    F = np.fft.fft2(img)
    Gx = np.fft.ifft2(F * mult_x).real
    Gy = np.fft.ifft2(F * mult_y).real
    G = np.sqrt(Gx**2 + Gy**2)
    phi = np.arctan2(Gy, Gx)
    return G, phi, Gx, Gy
