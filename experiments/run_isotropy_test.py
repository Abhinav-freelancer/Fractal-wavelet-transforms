"""
Test the specific claim made about the Riesz operator: that its response
is more isotropic (direction-independent) than the 4-directional G-L /
DF3SOp masks, since those combine discrete masks at only 0/45/90/135 deg.

Method: take one image, rotate it by a set of angles, apply each operator,
rotate the *response* back, and measure how much the response map changes
across rotations (it shouldn't, for a truly isotropic operator, other than
interpolation artifacts). We report the coefficient of variation of total
gradient energy across rotations -- lower is more isotropic.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from skimage import data
from skimage.transform import resize, rotate

from eval.degrade import to01
from core.gl_sobel import gl_sobel_gradient
from core.df3sop import df3sop_gradient
from core.riesz import riesz_magnitude

SIZE = 128
ANGLES = [0, 15, 30, 45, 60, 75, 90]


def energy_across_rotations(img01, operator_fn):
    energies = []
    for a in ANGLES:
        rotated = rotate(img01, a, mode="reflect")
        resp = operator_fn(rotated)
        energies.append(np.sum(resp**2))
    energies = np.array(energies)
    cv = np.std(energies) / (np.mean(energies) + 1e-8)
    return cv, energies


def main():
    img = resize(data.camera(), (SIZE, SIZE), anti_aliasing=True)
    clean01 = to01(img)

    operators = {
        "GL-FracSobel(nu=0.8)": lambda im: gl_sobel_gradient(im, nu=0.8)[0],
        "DF3SOp(alpha=1,beta=0.5)": lambda im: df3sop_gradient(im, 1.0, 0.5)[0],
        "Riesz(alpha=1.0)": lambda im: riesz_magnitude(im, 1.0),
    }

    print(f"{'Operator':<28}{'CV of energy across rotations':>32}")
    print("-" * 60)
    results = {}
    for name, fn in operators.items():
        cv, energies = energy_across_rotations(clean01, fn)
        results[name] = (cv, energies)
        print(f"{name:<28}{cv:>32.4f}")

    print("\nRaw per-angle total energies:")
    print(f"{'Operator':<28}" + "".join(f"{a:>10}deg" for a in ANGLES))
    for name, (cv, energies) in results.items():
        print(f"{name:<28}" + "".join(f"{e:>13.1f}" for e in energies))

    return results


if __name__ == "__main__":
    main()
