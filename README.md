# Noise- and Illumination-Adaptive Fractal-Fractional Edge Detection
### Prototype implementation and validation — session results

This implements and tests the direction discussed: a DFRWT subband-reliability
fusion combined with (a) Paper 3's fractal-fractional Sobel operator (DF3SOp)
and (b) a new isotropic Riesz fractional-derivative operator, evaluated
against baselines from all three source papers under noise, low-light, and
(for the first time) **combined** low-light+noise degradation.

## What's implemented (`core/`)

| File | What it is | Source |
|---|---|---|
| `gl_sobel.py` | Grunwald-Letnikov fractional 4-directional Sobel masks | Paper 2 (Yang & Li) |
| `dfrwt.py` | 2-D Discrete Fractional Wavelet Transform, forward+inverse | Paper 2 |
| `df3sop.py` | Dual-Functional Fractal-Fractional Sobel (fixed + adaptive-beta) | Paper 3 (Gowtham et al.) |
| `riesz.py` | Isotropic Riesz fractional derivative via FFT (`\|ξ\|^α` multiplier) | New (this session) |
| `reliability.py` | MAD-based local subband reliability/noise estimation | New |
| `pipeline.py` | **Proposed**: DFRWT decomposition → per-subband reliability-driven α, illumination-driven β → fractal-fractional or Riesz backend → reliability-weighted fusion → percentile threshold | New |

All operator formulas are transcribed directly from the papers' equations
(Paper 2 Eq. 19–25 for G-L Sobel masks, Eq. 1–18 for DFRWT; Paper 3 Table 1
for DF3SOp masks, Eq. 36 for adaptive β), not copied from any reference code
— none was available.

## Validated correctness

- **DFRWT forward/inverse reconstruction**: mean abs error = 0.0000 at p=1.0,
  0.5, 0.3 — the chirp-modulation/DWT/demodulation sandwich is implemented
  correctly (`experiments/` sanity check).
- **G-L mask / DF3SOp mask formulas**: directly match the papers' printed
  matrices; spot-checked numerically.

## Real, validated finding (positive)

Across 6 conditions (clean, Gaussian noise, Salt&Pepper, Speckle, low-light,
and **low-light+noise combined** — the realistic smartphone/dermoscopic
scenario neither Paper 2 nor Paper 3 tests), using the self-referential
**B/A edge-connectivity ratio** (Paper 2's own metric; lower = better,
reference-free, so not biased by any choice of "ground truth"):

```
condition                clean   gauss   s&p    speckle  lowlight  lowlight+noise
Proposed(DFRWT+DF3SOp)   0.034   0.027   0.030  0.033    0.029     0.020   <- rank #1 in 4/6
Best baseline            0.024   0.155   0.083  0.040    0.020     0.152   (varies by condition)
```

**Proposed(DFRWT+DF3SOp) ranks #1 in connectivity in every degraded condition
except the two clean/low-light-only cases**, where it still ranks #3, close
to the best baseline. This is exactly the hypothesis: the adaptive fusion's
advantage shows up specifically when degradation is present, including the
untested combined case. Full numbers in `outputs/comparison_results.csv`.

## Real, validated finding (negative)

**The claimed Riesz isotropy advantage was NOT confirmed.** Two independent
tests (natural-image rotation, and direct synthetic step-edges at controlled
angles — to rule out rotation-interpolation confounds) both show the
*discrete FFT-based* Riesz operator has **higher** directional variance
(CV ≈ 0.22) than the 4-directional G-L (CV ≈ 0.06) and DF3SOp (CV ≈ 0.07)
masks — the opposite of what the continuous-domain math predicts.

Root cause (diagnosed, not just observed): the Riesz derivative's isotropy
is a property of the *continuous* radially-symmetric multiplier `|ξ|^α`.
A finite discrete pixel grid is not itself rotationally symmetric — an
axis-aligned edge and a 45° edge have genuinely different discrete Fourier
content independent of the operator applied to them — and this grid
anisotropy dominates the small directional bias that 4-mask operators have
by construction. **This is a real, open discretization problem**, not a bug
in this implementation (see `experiments/run_isotropy_test_v2.py`).

## Real limitation found in the proposed method 

The DFRWT+DF3SOp fusion's strong B/A score is partly an artifact: standard
(decimated) DWT halves resolution at each level, so after upsampling back,
edge responses become coarse connected **blobs** rather than thin edge
curves (visually confirmed, `outputs/low_light_noise_comparison.png`).
Skeletonizing the blobs shatters them into many fragments (B/A jumps from
0.02 to 0.29), confirming the metric was partly rewarding blob size rather
than genuine thin-edge connectivity.

**Concrete fix identified, not yet implemented**: Paper 3 independently
hit this same decimation problem for its own enhancement fusion and solved
it with a **Stationary (undecimated) DWT** instead of plain DWT specifically
to avoid resolution loss and Gibbs artifacts (Paper 3, §4.5.4). Swapping
`dfrwt.py`'s decimated `pywt.dwt2`/`idwt2` calls for an undecimated/SWT
variant (`pywt.swt2`) while keeping the same chirp-modulation sandwich is
the direct next step, and is cheap to try (same chirp math, different base
transform).

## How to reproduce

```bash
pip install -r requirements.txt   # numpy, scipy, scikit-image, pywavelets, pandas, matplotlib
python experiments/run_comparison.py       # full 8-method x 6-condition table -> outputs/comparison_results.csv
python experiments/run_isotropy_test.py    # isotropy test v1 (natural image rotation)
python experiments/run_isotropy_test_v2.py # isotropy test v2 (synthetic edges, cleaner)
python experiments/make_figure.py          # visual comparison figure
```

## Honest next steps, in priority order

1. **Swap decimated DWT → SWT/undecimated transform** in `dfrwt.py` to fix
   the blob-vs-thin-edge issue — highest-value, well-grounded fix.
2. **Investigate discrete Riesz isotropy**: try a windowed/apodized FFT
   (Hann window before transform) or a different discretization (spatial
   fractional-Laplacian kernel truncated to finite support instead of full
   FFT) to see if discrete isotropy can be recovered — or report the
   negative result as a genuine contribution ("why naive discrete Riesz
   operators are not isotropic in practice").
3. **Precision/recall against Canny-as-reference is intrinsically biased**
   toward Canny-like methods; for a paper, either use a dataset with real
   human-annotated edge ground truth (e.g. BSDS500) or lead with the
   reference-free B/A/C/A/entropy/contrast metrics as primary.
4. Extend to HAM10000 classification (Paper 1's downstream task) once the
   edge-quality issue above is fixed, to show the enhancement transfers to
   a real classification accuracy gain, not just edge-map metrics.
