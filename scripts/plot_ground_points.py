import laspy
import numpy as np
import pyvista as pv

LAZ_PATH = "data/raw/fort_washington_sample.laz"
SAMPLE_THRESHOLD = 1_000_000
SAMPLE_FRACTION = 0.10

print("Reading point cloud...")
las = laspy.read(LAZ_PATH)

ground_mask = las.classification == 2
total_ground = ground_mask.sum()
print(f"Ground points: {total_ground:,}")

x = np.array(las.x[ground_mask])
y = np.array(las.y[ground_mask])
z = np.array(las.z[ground_mask])
r = np.array(las.red[ground_mask])
g = np.array(las.green[ground_mask])
b = np.array(las.blue[ground_mask])

if total_ground > SAMPLE_THRESHOLD:
    n_sample = int(total_ground * SAMPLE_FRACTION)
    print(f"Sampling {n_sample:,} points ({SAMPLE_FRACTION*100:.0f}%)...")
    rng = np.random.default_rng(42)
    idx = rng.choice(total_ground, size=n_sample, replace=False)
    x, y, z, r, g, b = x[idx], y[idx], z[idx], r[idx], g[idx], b[idx]

# Center coordinates to avoid floating point precision issues
x = x - x.mean()
y = y - y.mean()

points = np.column_stack([x, y, z])

# LAS stores RGB as 16-bit; normalize to [0, 1]
rgb = np.column_stack([r, g, b]).astype(np.float32) / 65535.0

cloud = pv.PolyData(points)
cloud["RGB"] = (rgb * 255).astype(np.uint8)

print(f"Rendering {len(points):,} points...")

pl = pv.Plotter()
pl.add_points(
    cloud,
    scalars="RGB",
    rgb=True,
    point_size=1.5,
    render_points_as_spheres=False,
)
pl.add_text(
    f"Fort Washington — Ground Points\n{len(points):,} points ({SAMPLE_FRACTION*100:.0f}% sample)",
    font_size=10,
)
pl.show_axes()
pl.show()
