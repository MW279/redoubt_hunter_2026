import laspy
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

LAZ_PATH = "data/raw/fort_washington_sample.laz"
SAMPLE_THRESHOLD = 1_000_000
SAMPLE_FRACTION = 0.10
K_NEIGHBORS = 20  # local neighborhood size for plane fitting

print("Reading point cloud...")
las = laspy.read(LAZ_PATH)

ground_mask = las.classification == 2
total_ground = ground_mask.sum()
print(f"Ground points: {total_ground:,}")

x = np.array(las.x[ground_mask])
y = np.array(las.y[ground_mask])
z = np.array(las.z[ground_mask])

if total_ground > SAMPLE_THRESHOLD:
    n_sample = int(total_ground * SAMPLE_FRACTION)
    print(f"Sampling {n_sample:,} points ({SAMPLE_FRACTION*100:.0f}%)...")
    rng = np.random.default_rng(42)
    idx = rng.choice(total_ground, size=n_sample, replace=False)
    x, y, z = x[idx], y[idx], z[idx]

x = x - x.mean()
y = y - y.mean()
points = np.column_stack([x, y, z])
n_points = len(points)

# ── Slope via local PCA ───────────────────────────────────────────────────────
# For each point: find K neighbors → fit local plane → angle of normal from
# vertical = slope.  eigh() on [N,3,3] covariance matrices is fully vectorised.

print(f"Building KD-tree on {n_points:,} points...")
tree = KDTree(points)

print(f"Querying {K_NEIGHBORS} nearest neighbours for every point...")
_, indices = tree.query(points, k=K_NEIGHBORS + 1)  # +1 includes self
indices = indices[:, 1:]                              # drop self-match

print("Fitting local planes and computing slope angles...")
neighborhoods = points[indices]                           # [N, K, 3]
centroids     = neighborhoods.mean(axis=1, keepdims=True) # [N, 1, 3]
centered      = neighborhoods - centroids                 # [N, K, 3]

# Covariance matrix per point
covs = np.einsum("nki,nkj->nij", centered, centered) / K_NEIGHBORS  # [N, 3, 3]

# eigh returns eigenvalues ascending; eigenvec of smallest eigenvalue = surface normal
_, eigvecs = np.linalg.eigh(covs)   # eigvecs: [N, 3, 3], columns are vectors
normals = eigvecs[:, :, 0]          # [N, 3] — normal for each point

# Slope = angle between normal and vertical Z axis, in degrees
slope_deg = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0)))

p2, p98 = np.percentile(slope_deg, [2, 98])
print(f"Slope  min={slope_deg.min():.1f}°  max={slope_deg.max():.1f}°  "
      f"mean={slope_deg.mean():.1f}°  (2–98 pct: {p2:.1f}°–{p98:.1f}°)")

# ── Visualise ─────────────────────────────────────────────────────────────────
print(f"Rendering {n_points:,} points...")

cloud = pv.PolyData(points)
cloud["Slope (degrees)"] = slope_deg

pl = pv.Plotter()
pl.set_background("black")

actor = pl.add_points(
    cloud,
    scalars="Slope (degrees)",
    cmap="inferno",        # dark=flat, bright yellow/white=steep walls
    clim=[p2, p98],        # clip colour range to avoid outlier washout
    point_size=1.5,
    render_points_as_spheres=False,
)
pl.add_scalar_bar(
    "Slope (°)",
    vertical=True,
    fmt="%.0f",
    color="white",
)
pl.add_text(
    f"Fort Washington — Ground Slope\n"
    f"{n_points:,} points  |  K={K_NEIGHBORS} neighbours  |  "
    f"{SAMPLE_FRACTION*100:.0f}% sample",
    font_size=9,
    color="white",
)
pl.show_axes()
pl.show()
