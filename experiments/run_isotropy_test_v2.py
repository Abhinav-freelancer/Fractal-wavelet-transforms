"""
Follow-up isotropy test addressing a confound in v1: skimage.rotate()
on a natural photo changes the border content (reflect-padding corners)
every time, and FFT-domain operators (Riesz) are sensitive to the
resulting boundary discontinuities under FFT's implicit periodicity
assumption. This version instead synthesizes a single straight edge
directly at a controlled angle (no interpolation, no new boundary
artifacts introduced by rotation), isolating the operator's true
directional response from boundary-handling artifacts.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.gl_sobel import gl_sobel_gradient
from core.df3sop import df3sop_gradient
from core.riesz import riesz_magnitude

SIZE = 128
ANGLES = [0, 15, 30, 45, 60, 75, 90]


def synthetic_edge(size, angle_deg):
    """A clean step edge through the image center at the given angle."""
    y, x = np.mgrid[0:size, 0:size]
    cx = cy = size / 2
    theta = np.deg2rad(angle_deg)
    # signed distance from the line through (cx,cy) with normal (cos,sin)
    d = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
    img = (d > 0).astype(np.float64)
    return img


def response_energy_near_edge(resp, size, band=10):
    """Sum response energy in a band around the image center (where the
    synthetic edge actually is), avoiding border effects far from it."""
    c = size // 2
    region = resp[c - band:c + band, c - band:c + band]
    return np.sum(region**2)


def main():
    operators = {
        "GL-FracSobel(nu=0.8)": lambda im: gl_sobel_gradient(im, nu=0.8)[0],
        "DF3SOp(alpha=1,beta=0.5)": lambda im: df3sop_gradient(im, 1.0, 0.5)[0],
        "Riesz(alpha=1.0)": lambda im: riesz_magnitude(im, 1.0),
    }

    print(f"{'Operator':<28}{'CV of edge-local energy across angles':>40}")
    print("-" * 70)
    all_results = {}
    for name, fn in operators.items():
        energies = []
        for a in ANGLES:
            img = synthetic_edge(SIZE, a)
            resp = fn(img)
            e = response_energy_near_edge(resp, SIZE, band=12)
            energies.append(e)
        energies = np.array(energies)
        cv = np.std(energies) / (np.mean(energies) + 1e-8)
        all_results[name] = energies
        print(f"{name:<28}{cv:>40.4f}")

    print("\nRaw per-angle edge-local energies:")
    print(f"{'Operator':<28}" + "".join(f"{a:>10}deg" for a in ANGLES))
    for name, energies in all_results.items():
        print(f"{name:<28}" + "".join(f"{e:>13.1f}" for e in energies))


if __name__ == "__main__":
    main()
