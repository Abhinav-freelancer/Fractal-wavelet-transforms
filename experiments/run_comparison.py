import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from skimage import data
from skimage.transform import resize
from skimage.feature import canny

from eval.degrade import to01, add_gaussian, add_salt_pepper, add_speckle, low_light, low_light_plus_noise
from eval.metrics import ssim, mse, entropy, michelson_contrast, edge_density, connectivity_ratio, precision_recall_f1

from core.baselines import (
    method_canny, method_sobel, method_gl_fractional_sobel,
    method_df3sop_fixed, method_df3sop_adaptive_beta, method_riesz_fixed,
)
from core.pipeline import proposed_edge_map

SIZE = 160  # keep runtime reasonable across many conditions/methods

METHODS = {
    "Canny":                lambda im: method_canny(im),
    "Sobel":                lambda im: method_sobel(im),
    "GL-FracSobel(nu=.8)":  lambda im: method_gl_fractional_sobel(im, nu=0.8),
    "DF3SOp-fixed":         lambda im: method_df3sop_fixed(im, alpha=1.0, beta=0.5),
    "DF3SOp-adaptive-beta": lambda im: method_df3sop_adaptive_beta(im, alpha=1.0),
    "Riesz-fixed(a=1.0)":   lambda im: method_riesz_fixed(im, alpha=1.0),
    "Proposed(DFRWT+DF3SOp)": lambda im: proposed_edge_map(im, base_p=0.5, backend="df3sop"),
    "Proposed(DFRWT+Riesz)":  lambda im: proposed_edge_map(im, base_p=0.5, backend="riesz"),
}


def build_conditions(clean01):
    return {
        "clean":            clean01,
        "gaussian_noise":   add_gaussian(clean01, var=0.01, seed=1),
        "salt_pepper":      add_salt_pepper(clean01, amount=0.05, seed=1),
        "speckle":          add_speckle(clean01, var=0.02, seed=1),
        "low_light":        low_light(clean01, gamma=2.2, scale=0.35),
        "low_light+noise":  low_light_plus_noise(clean01, gamma=2.2, scale=0.35, noise_var=0.004, seed=1),
    }


def main():
    img = data.camera()
    img = resize(img, (SIZE, SIZE), anti_aliasing=True)
    clean01 = to01(img)
    ref_binary = canny(clean01, sigma=1.0)  # ground-truth-proxy edge map from the clean image

    conditions = build_conditions(clean01)
    rows = []

    for cond_name, degraded in conditions.items():
        for method_name, fn in METHODS.items():
            t0 = time.time()
            soft, binary = fn(degraded)
            dt = time.time() - t0

            A, B, C, b_over_a, c_over_a = connectivity_ratio(binary)
            prec, rec, f1 = precision_recall_f1(binary, ref_binary, tolerance=2)

            rows.append({
                "condition": cond_name,
                "method": method_name,
                "SSIM_vs_clean_img": ssim(degraded, clean01),
                "edge_density": edge_density(binary),
                "B/A": b_over_a,
                "C/A": c_over_a,
                "precision": prec,
                "recall": rec,
                "F1_vs_clean_edges": f1,
                "entropy": entropy(soft),
                "contrast": michelson_contrast(degraded),
                "time_sec": dt,
            })
        print(f"done: {cond_name}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(__file__), "..", "outputs", "comparison_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}")
    return df


if __name__ == "__main__":
    df = main()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.round(4))
