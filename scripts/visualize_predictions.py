import laspy
import numpy as np
import pyvista as pv

las = laspy.read("data/processed/predictions_01.laz")

x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)
pred = np.array(las.pred_class).astype(int)

x -= x.mean()
y -= y.mean()

points = np.column_stack([x, y, z])
ground_mask = pred == 0
wall_mask   = pred == 1

n_ground = ground_mask.sum()
n_wall   = wall_mask.sum()
print(f"Ground: {n_ground:,}  |  Predicted wall: {n_wall:,}  |  Total: {len(points):,}")

ground_cloud = pv.PolyData(points[ground_mask])
wall_cloud   = pv.PolyData(points[wall_mask])

pl = pv.Plotter()
pl.set_background("black")

pl.add_points(
    ground_cloud,
    color="#3a4a6b",      # dark slate-blue
    opacity=0.25,
    point_size=1.5,
    render_points_as_spheres=False,
    label=f"Ground ({n_ground:,})",
)
pl.add_points(
    wall_cloud,
    color="#ff3300",      # bright red
    opacity=1.0,
    point_size=3.0,
    render_points_as_spheres=False,
    label=f"Predicted wall ({n_wall:,})",
)

pl.add_legend(bcolor=(0.1, 0.1, 0.1), border=True, size=(0.22, 0.10))
pl.add_text(
    "SE(3)-GNN Predictions — Fort Washington Redoubt\n"
    "Red = predicted wall  |  Blue (transparent) = ground",
    font_size=9,
    color="white",
)
pl.show_axes()
pl.show()
