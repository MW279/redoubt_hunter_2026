import laspy
import numpy as np
import matplotlib.pyplot as plt

LAZ_PATH = "data/raw/fort_washington_sample.laz"

las = laspy.read(LAZ_PATH)
x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)

print(f"X: {x.min():.2f} → {x.max():.2f}  (range {x.max()-x.min():.1f} m)")
print(f"Y: {y.min():.2f} → {y.max():.2f}  (range {y.max()-y.min():.1f} m)")
print(f"Z: {z.min():.2f} → {z.max():.2f}")

# Subsample for plotting — every 10th point (~445K points)
step = 10
xs, ys, zs = x[::step], y[::step], z[::step]
print(f"Plotting {len(xs):,} points...")

vmin, vmax = np.percentile(zs, 2), np.percentile(zs, 98)

fig, ax = plt.subplots(figsize=(12, 13))
sc = ax.scatter(xs, ys, c=zs, s=0.3, cmap="terrain", vmin=vmin, vmax=vmax, linewidths=0)
plt.colorbar(sc, ax=ax, label="Elevation (m)", shrink=0.6)

# Grid lines every 100 m
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
for xg in np.arange(np.ceil(x_min / 100) * 100, x_max, 100):
    ax.axvline(xg, color="gray", lw=0.4, alpha=0.5)
for yg in np.arange(np.ceil(y_min / 100) * 100, y_max, 100):
    ax.axhline(yg, color="gray", lw=0.4, alpha=0.5)

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_title("Fort Washington — Elevation overview\n"
             "Hover over the redoubt corners → coordinates print in terminal")
ax.tick_params(labelsize=7)

def on_move(event):
    if event.inaxes:
        print(f"\rX={event.xdata:.1f}  Y={event.ydata:.1f}", end="", flush=True)

fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.tight_layout()
plt.show()
print()
