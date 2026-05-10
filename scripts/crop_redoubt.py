import laspy
import numpy as np

LAZ_PATH  = "data/raw/fort_washington_sample.laz"
OUT_PATH  = "data/processed/training_sample_01.laz"
BUFFER    = 20.0  # metres padding beyond the selected corners

# Corners picked from the overview map
corners_x = [322988.2, 323013.1, 323118.9, 323216.1]
corners_y = [4286675.2, 4286578.4, 4286776.0, 4286723.2]

x_min = min(corners_x) - BUFFER
x_max = max(corners_x) + BUFFER
y_min = min(corners_y) - BUFFER
y_max = max(corners_y) + BUFFER

print(f"Bounding box (with {BUFFER} m buffer):")
print(f"  X: {x_min:.2f} → {x_max:.2f}  ({x_max - x_min:.1f} m)")
print(f"  Y: {y_min:.2f} → {y_max:.2f}  ({y_max - y_min:.1f} m)")

print("\nReading source file...")
las = laspy.read(LAZ_PATH)

x = np.array(las.x)
y = np.array(las.y)

mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
n_kept = mask.sum()
print(f"Points inside box: {n_kept:,} of {len(x):,} total")

cropped = las[mask]

cropped.write(OUT_PATH)
print(f"\nSaved → {OUT_PATH}")

# Sanity check
check = laspy.read(OUT_PATH)
print(f"Verified: {len(check.points):,} points in saved file")
print(f"  X: {check.x.min():.2f} → {check.x.max():.2f}")
print(f"  Y: {check.y.min():.2f} → {check.y.max():.2f}")
print(f"  Z: {check.z.min():.2f} → {check.z.max():.2f}")
