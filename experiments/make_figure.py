import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize

from eval.degrade import to01, low_light_plus_noise, add_gaussian
from core.baselines import method_canny, method_sobel, method_gl_fractional_sobel, method_riesz_fixed
from core.pipeline import proposed_edge_map

SIZE = 160
img = resize(data.camera(), (SIZE, SIZE), anti_aliasing=True)
clean01 = to01(img)
degraded = low_light_plus_noise(clean01, gamma=2.2, scale=0.35, noise_var=0.004, seed=1)

methods = [
    ("Input (low-light + noise)", None),
    ("Canny", method_canny),
    ("Sobel", method_sobel),
    ("GL-FracSobel (nu=0.8)", lambda im: method_gl_fractional_sobel(im, nu=0.8)),
    ("Riesz-fixed (a=1.0)", lambda im: method_riesz_fixed(im, alpha=1.0)),
    ("Proposed (DFRWT+DF3SOp)", lambda im: proposed_edge_map(im, base_p=0.5, backend="df3sop")),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
axes = axes.ravel()

for ax, (name, fn) in zip(axes, methods):
    if fn is None:
        ax.imshow(degraded, cmap="gray")
    else:
        _, binary = fn(degraded)
        ax.imshow(binary, cmap="gray")
    ax.set_title(name, fontsize=11)
    ax.axis("off")

plt.suptitle("Edge detection under combined low-light + sensor noise\n"
             "(untested scenario in prior papers)", fontsize=13)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "low_light_noise_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved figure to {out_path}")
