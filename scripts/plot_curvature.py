import laspy
import numpy as np
import pyvista as pv

las = laspy.read("data/processed/features_01.laz")

x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)
curvature = np.array(las.curvature)

x -= x.mean()
y -= y.mean()

points = np.column_stack([x, y, z])

p2, p98 = np.percentile(curvature, [2, 98])
print(f"Curvature  min={curvature.min():.6f}  max={curvature.max():.6f}  "
      f"mean={curvature.mean():.6f}  (2–98 pct: {p2:.6f}–{p98:.6f})")
print(f"Rendering {len(points):,} points...")

cloud = pv.PolyData(points)
cloud["Curvature"] = curvature

pl = pv.Plotter()
pl.set_background("black")
pl.add_points(
    cloud,
    scalars="Curvature",
    cmap="inferno",
    clim=[p2, p98],
    point_size=2.5,
    render_points_as_spheres=False,
)
pl.add_scalar_bar("Curvature", vertical=True, fmt="%.4f", color="white")
pl.add_text(
    f"Fort Washington Redoubt — Surface Curvature\n{len(points):,} points  |  K=30 neighbours",
    font_size=9,
    color="white",
)
pl.show_axes()
pl.show()
