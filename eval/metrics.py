"""
Evaluation metrics reproduced from the three source papers so results are
directly comparable to their tables.

- SSIM, MSE: standard (skimage)
- entropy: Shannon entropy of pixel intensity histogram (Paper 3, Eq 35)
- michelson_contrast: (Imax-Imin)/(Imax+Imin) (Paper 3)
- edge_density: fraction of pixels marked as edge (Paper 3, Table 6)
- connectivity_ratio: B/A, C/A from Paper 2 Table 1 -- A = total edge
  points, B = # 4-connected components, C = # 8-connected components.
  Smaller ratio = better edge connectivity/integrity.
- precision_recall_f1: against a reference edge map (not in the papers,
  but needed since we use synthetic noise so we *do* have a clean-image
  reference available, which the papers mostly lack)
"""
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import mean_squared_error as sk_mse
from scipy.ndimage import label


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    return float(sk_ssim(a, b, data_range=1.0))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    return float(sk_mse(a, b))


def entropy(img: np.ndarray, bins: int = 256) -> float:
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    hist, _ = np.histogram(img, bins=bins, range=(0, 1), density=False)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def michelson_contrast(img: np.ndarray) -> float:
    imax, imin = img.max(), img.min()
    return float((imax - imin) / (imax + imin + 1e-8))


def edge_density(binary_edge_map: np.ndarray) -> float:
    return float(np.count_nonzero(binary_edge_map) / binary_edge_map.size)


def connectivity_ratio(binary_edge_map: np.ndarray):
    """Returns (A, B, C, B/A, C/A) per Paper 2's definition."""
    edge_map = binary_edge_map.astype(bool)
    A = int(np.count_nonzero(edge_map))
    if A == 0:
        return 0, 0, 0, 0.0, 0.0
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    struct8 = np.ones((3, 3))
    _, nB = label(edge_map, structure=struct4)
    _, nC = label(edge_map, structure=struct8)
    return A, nB, nC, nB / A, nC / A


def precision_recall_f1(pred_binary: np.ndarray, ref_binary: np.ndarray, tolerance: int = 2):
    """Edge PR/F1 with a small dilation tolerance (standard practice for
    edge-map comparison, since pixel-exact overlap is overly strict)."""
    from scipy.ndimage import binary_dilation
    pred = pred_binary.astype(bool)
    ref = ref_binary.astype(bool)
    ref_dil = binary_dilation(ref, iterations=tolerance)
    pred_dil = binary_dilation(pred, iterations=tolerance)

    tp = np.count_nonzero(pred & ref_dil)
    fp = np.count_nonzero(pred & ~ref_dil)
    fn = np.count_nonzero(ref & ~pred_dil)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def isotropy_score(responses_by_angle: dict) -> float:
    """Novel metric: coefficient of variation of gradient-magnitude energy
    across rotated versions of the same image. Lower = more isotropic
    (direction-independent) response. Used to validate the Riesz operator's
    claimed advantage over 4-directional masks.
    """
    energies = np.array([np.sum(r**2) for r in responses_by_angle.values()])
    return float(np.std(energies) / (np.mean(energies) + 1e-8))
