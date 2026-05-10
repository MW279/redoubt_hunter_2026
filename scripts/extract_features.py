import laspy
import numpy as np
from scipy.spatial import KDTree

LAZ_PATH = "data/processed/training_sample_01.laz"
OUT_PATH = "data/processed/features_01.laz"
K = 30

print("Reading point cloud...")
las = laspy.read(LAZ_PATH)
x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)
n_points = len(x)
print(f"  {n_points:,} points loaded")

points = np.column_stack([x, y, z])

# ── KNN ───────────────────────────────────────────────────────────────────────
print(f"Building KD-tree...")
tree = KDTree(points)

print(f"Querying {K} nearest neighbours for every point...")
_, indices = tree.query(points, k=K + 1)  # +1 because query includes self
indices = indices[:, 1:]                   # drop self-match → shape [N, K]

# ── Local PCA ────────────────────────────────────────────────────────────────
print("Running PCA on local neighbourhoods...")
neighborhoods = points[indices]                            # [N, K, 3]
centroids     = neighborhoods.mean(axis=1, keepdims=True)  # [N, 1, 3]
centered      = neighborhoods - centroids                  # [N, K, 3]

# Covariance matrix per point
covs = np.einsum("nki,nkj->nij", centered, centered) / K  # [N, 3, 3]

# eigh returns eigenvalues in ascending order (λ0 ≤ λ1 ≤ λ2)
eigenvalues, eigenvectors = np.linalg.eigh(covs)

# Normal = eigenvector of the smallest eigenvalue (most compressed axis = surface normal)
normals = eigenvectors[:, :, 0]  # [N, 3]

# Orient all normals to point upward (+Z hemisphere) for consistency
flip_mask = normals[:, 2] < 0
normals[flip_mask] *= -1

# Surface curvature = λ0 / (λ0 + λ1 + λ2)
# Fraction of variance along the normal direction: 0 = flat plane, ~0.33 = isotropic noise
total_variance = eigenvalues.sum(axis=1)
curvature = eigenvalues[:, 0] / np.where(total_variance > 0, total_variance, 1.0)

nx        = normals[:, 0].astype(np.float32)
ny        = normals[:, 1].astype(np.float32)
nz        = normals[:, 2].astype(np.float32)
curvature = curvature.astype(np.float32)

print(f"\n  nx        : {nx.min():.4f} → {nx.max():.4f}")
print(f"  ny        : {ny.min():.4f} → {ny.max():.4f}")
print(f"  nz        : {nz.min():.4f} → {nz.max():.4f}")
print(f"  curvature : min={curvature.min():.6f}  max={curvature.max():.6f}  "
      f"mean={curvature.mean():.6f}  p99={np.percentile(curvature, 99):.6f}")

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"\nWriting {OUT_PATH}...")
out = laspy.LasData(header=las.header)
out.points = las.points.copy()

out.add_extra_dims([
    laspy.ExtraBytesParams(name="nx",        type=np.float32, description="Normal X"),
    laspy.ExtraBytesParams(name="ny",        type=np.float32, description="Normal Y"),
    laspy.ExtraBytesParams(name="nz",        type=np.float32, description="Normal Z"),
    laspy.ExtraBytesParams(name="curvature", type=np.float32, description="Surface curvature (PCA)"),
])

out.nx        = nx
out.ny        = ny
out.nz        = nz
out.curvature = curvature

out.write(OUT_PATH)

# ── Verify ────────────────────────────────────────────────────────────────────
check = laspy.read(OUT_PATH)
print(f"\nSaved → {OUT_PATH}")
print(f"  Points    : {len(check.points):,}")
print(f"  Extra dims: {list(check.point_format.extra_dim_names)}")
print(f"  nx sample : {np.array(check.nx[:5]).round(4)}")
print(f"  curvature sample: {np.array(check.curvature[:5]).round(6)}")
