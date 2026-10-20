"""Generate a small synthetic dataset so the project runs without satellite downloads."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "demo_patches.npz"
CLASS_NAMES = ["smallholder_agriculture", "large_scale_agriculture", "road",
               "selective_logging", "mining"]

def draw_patch(label, rng, size=32):
    # Four illustrative channels: VV, VH, NIR, RED. These are simulated patterns,
    # not calibrated reflectance/backscatter and not actual Sentinel measurements.
    yy, xx = np.mgrid[0:size, 0:size]
    patch = np.zeros((size, size, 4), dtype=np.float32)
    for c, base in enumerate([0.45, 0.35, 0.65, 0.25]):
        patch[..., c] = base + rng.normal(0, 0.035, (size, size))
    if label == 0:  # small irregular clearings
        for _ in range(5):
            cx, cy = rng.integers(3, size-3, size=2)
            rx, ry = rng.integers(2, 5, size=2)
            mask = ((xx-cx)/rx)**2 + ((yy-cy)/ry)**2 < 1
            patch[mask, 0] -= .15; patch[mask, 1] -= .10
            patch[mask, 2] -= .35; patch[mask, 3] += .20
    elif label == 1:  # large rectangular clearing
        x0, y0 = rng.integers(2, 10, size=2)
        x1, y1 = rng.integers(22, 30, size=2)
        patch[y0:y1, x0:x1, 0] -= .12
        patch[y0:y1, x0:x1, 1] -= .10
        patch[y0:y1, x0:x1, 2] -= .38
        patch[y0:y1, x0:x1, 3] += .22
    elif label == 2:  # narrow linear road
        offset = int(rng.integers(-5, 6))
        line = np.abs(yy - (xx + offset)) <= 1
        patch[line, 0] += .22; patch[line, 1] += .18
        patch[line, 2] -= .25; patch[line, 3] += .10
    elif label == 3:  # small scattered logging gaps
        for _ in range(10):
            cx, cy = rng.integers(1, size-1, size=2)
            patch[max(0,cy-1):cy+2, max(0,cx-1):cx+2, 0] -= .12
            patch[max(0,cy-1):cy+2, max(0,cx-1):cx+2, 2] -= .20
    else:  # mining: bright central irregular zone
        cx, cy = rng.integers(10, 22, size=2)
        radius = rng.integers(5, 10)
        mask = (xx-cx)**2 + (yy-cy)**2 < radius**2
        patch[mask, 0] += .28; patch[mask, 1] += .20
        patch[mask, 2] -= .42; patch[mask, 3] += .28
    return np.clip(patch, 0, 1)

def main():
    rng = np.random.default_rng(42)
    per_class = 160
    X, y = [], []
    for label in range(len(CLASS_NAMES)):
        for _ in range(per_class):
            X.append(draw_patch(label, rng))
            y.append(label)
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
    order = rng.permutation(len(y))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X[order], y=y[order])
    print(f"Saved {len(y)} synthetic patches to: {OUT}")
    print("Classes:", ", ".join(CLASS_NAMES))
    print("WARNING: synthetic educational data only; not real satellite imagery.")

if __name__ == "__main__":
    main()
