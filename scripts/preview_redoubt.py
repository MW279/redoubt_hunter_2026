import laspy
import numpy as np
import pyvista as pv

las = laspy.read("data/processed/training_sample_01.laz")

x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)
r = np.array(las.red)
g = np.array(las.green)
b = np.array(las.blue)

x -= x.mean()
y -= y.mean()

points = np.column_stack([x, y, z])
rgb = np.column_stack([r, g, b]).astype(np.float32) / 65535.0

cloud = pv.PolyData(points)
cloud["RGB"] = (rgb * 255).astype(np.uint8)
cloud["Elevation"] = z

pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

pl.subplot(0, 0)
pl.add_text("RGB Colour", font_size=10, color="white")
pl.add_points(cloud, scalars="RGB", rgb=True, point_size=2.5, render_points_as_spheres=False)
pl.set_background("black")
pl.show_axes()

pl.subplot(0, 1)
pl.add_text("Elevation", font_size=10, color="white")
pl.add_points(cloud, scalars="Elevation", cmap="terrain", point_size=2.5, render_points_as_spheres=False)
pl.add_scalar_bar("Elevation (m)", color="white", fmt="%.1f")
pl.set_background("black")
pl.show_axes()

pl.link_views()
pl.show()
