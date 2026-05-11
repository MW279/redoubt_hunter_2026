"""
SE(3)-Equivariant GNN for archaeological earthwork detection.

Architecture: equivariant message passing via e3nn tensor products.
  - Node features : curvature (0e scalar) + surface normal (1o vector)
  - Edge features : spherical harmonics of relative position up to L=2
  - Radial weights: MLP(distance) → tensor-product weights  (not equivariant,
                    but only depends on the invariant distance → whole layer IS equivariant)
  - Output        : per-point wall probability (SE(3)-invariant scalar)

Pseudo-labels: top-10% curvature → "wall/edge" (1), rest → "flat ground" (0).
"""

import time
import laspy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
from e3nn import o3
import pyvista as pv

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/processed/features_01.laz"
OUT_PATH      = "data/processed/predictions_01.laz"

RADIUS        = 0.8    # graph edge radius in metres
MAX_NEIGHBORS = 16     # cap neighbours per node
LMAX          = 2      # spherical harmonics up to L=2  →  1x0e + 1x1o + 1x2e
HIDDEN_MUL    = 8      # channel multiplicity in hidden layers
N_LAYERS      = 3      # number of equivariant message-passing layers
N_EPOCHS      = 60
LR            = 3e-3
EDGE_BATCH    = 500_000  # process edges in chunks to bound peak memory
CURVATURE_PCT = 93       # percentile above which a point is pseudo-labelled "wall"

# e3nn tensor-product kernels are CPU-optimised; MPS hits buffer-size limits
device = torch.device("cpu")
print(f"Device: {device}\n")


# ── 1. Load features ──────────────────────────────────────────────────────────
print("Loading features_01.laz ...")
las       = laspy.read(FEATURES_PATH)
pos_np    = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
normal_np = np.column_stack([las.nx, las.ny, las.nz]).astype(np.float32)
curv_np   = np.array(las.curvature, dtype=np.float32)
N         = len(pos_np)
print(f"  {N:,} points")

# Pseudo-labels
threshold   = float(np.percentile(curv_np, CURVATURE_PCT))
labels_np   = (curv_np > threshold).astype(np.int64)
n_wall      = labels_np.sum()
print(f"  Curvature threshold (p{CURVATURE_PCT}): {threshold:.6f}")
print(f"  Pseudo-labels — ground: {N - n_wall:,}  wall: {n_wall:,}")


# ── 2. Build radius graph with scipy KDTree ───────────────────────────────────
print(f"\nBuilding radius graph (r={RADIUS} m, max {MAX_NEIGHBORS} neighbours) ...")
tree = KDTree(pos_np)
indices = tree.query_ball_point(pos_np, r=RADIUS, workers=-1)

src_list, dst_list = [], []
for i, nbrs in enumerate(indices):
    nbrs = [j for j in nbrs if j != i]
    if len(nbrs) > MAX_NEIGHBORS:
        # Keep closest MAX_NEIGHBORS
        dists = np.linalg.norm(pos_np[nbrs] - pos_np[i], axis=1)
        nbrs  = [nbrs[k] for k in np.argsort(dists)[:MAX_NEIGHBORS]]
    for j in nbrs:
        src_list.append(i)
        dst_list.append(j)

src = torch.tensor(src_list, dtype=torch.long)
dst = torch.tensor(dst_list, dtype=torch.long)
print(f"  {len(src):,} edges  ({len(src)/N:.1f} per node)")

# Edge vectors and spherical harmonics
pos_t    = torch.tensor(pos_np)
edge_vec = pos_t[dst] - pos_t[src]                    # [E, 3]  (relative position)
edge_dist = edge_vec.norm(dim=-1, keepdim=True)        # [E, 1]  (invariant distance)
edge_dist_norm = edge_dist / RADIUS                    # normalise to [0,1]

irreps_sh = o3.Irreps.spherical_harmonics(lmax=LMAX)  # "1x0e + 1x1o + 1x2e"
edge_sh   = o3.spherical_harmonics(
    irreps_sh, edge_vec, normalize=True, normalization="component"
)  # [E, 9]


# ── 3. Node feature tensor ─────────────────────────────────────────────────────
# Layout must match irreps_in = "1x0e + 1x1o"
#   dim 0   : curvature  (type-0e scalar)
#   dims 1-3: normal xyz (type-1o vector, odd parity = pseudovector)
node_feat = torch.cat([
    torch.tensor(curv_np).unsqueeze(-1),
    torch.tensor(normal_np),
], dim=-1)  # [N, 4]

irreps_in = o3.Irreps("1x0e + 1x1o")
assert node_feat.shape[-1] == irreps_in.dim


# ── 4. Model ──────────────────────────────────────────────────────────────────
class SE3Layer(nn.Module):
    """
    One SE(3)-equivariant message-passing step.

    For every edge i→j:
      1. Compute spherical harmonics of r_ij  (equivariant, precomputed)
      2. Radial MLP(|r_ij|) → tensor-product weights  (depends only on distance)
      3. message_ij = TensorProduct(h_j, sh_ij, weights)
    Aggregate messages at each node (mean), add equivariant skip connection.
    No element-wise activation is applied to equivariant features — the
    tensor product itself provides the non-linearity.
    """
    def __init__(self, irreps_in, irreps_sh, irreps_out):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in, irreps_sh, irreps_out, shared_weights=False,
        )
        # Radial network: distance → weights for the tensor product
        self.radial = nn.Sequential(
            nn.Linear(1, 32),  nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, self.tp.weight_numel),
        )
        self.skip = o3.Linear(irreps_in, irreps_out)

    def forward(self, x, src, dst, edge_sh, edge_dist_norm, edge_batch=500_000):
        # Process edges in chunks so the [E, weight_numel] tensor stays small
        E       = src.shape[0]
        out_dim = self.tp.irreps_out.dim
        chunks  = []
        for s in range(0, E, edge_batch):
            e   = min(s + edge_batch, E)
            w   = self.radial(edge_dist_norm[s:e])           # [batch, weight_numel]
            m   = self.tp(x[src[s:e]], edge_sh[s:e], w)     # [batch, out_dim]
            chunks.append(m)
        messages = torch.cat(chunks, dim=0)                  # [E, out_dim]

        # Mean aggregation (equivariant — linear in messages)
        idx = dst.unsqueeze(-1).expand_as(messages)
        agg = torch.zeros(x.shape[0], out_dim, device=x.device).scatter_add(0, idx, messages)
        cnt = torch.zeros(x.shape[0], 1, device=x.device).scatter_add(
            0, dst.unsqueeze(-1), torch.ones(E, 1, device=x.device)
        )
        agg = agg / cnt.clamp(min=1)

        return agg + self.skip(x)   # residual skip


class SE3GNN(nn.Module):
    def __init__(self, irreps_in, irreps_sh, hidden_mul, n_layers, n_classes):
        super().__init__()
        # Hidden irreps: scalars + vectors + rank-2 tensors
        irreps_h = o3.Irreps(
            f"{hidden_mul}x0e + {hidden_mul // 2}x1o + {hidden_mul // 4}x2e"
        )
        # Lifting: map input irreps → hidden irreps
        self.lift = o3.Linear(irreps_in, irreps_h)

        self.layers = nn.ModuleList([
            SE3Layer(irreps_h, irreps_sh, irreps_h) for _ in range(n_layers)
        ])

        # Readout: project to scalars only (rotation-invariant), then classify
        n_scalars = hidden_mul
        self.to_scalars = o3.Linear(irreps_h, o3.Irreps(f"{n_scalars}x0e"))
        self.classifier  = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_scalars, 32),
            nn.SiLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, src, dst, edge_sh, edge_dist_norm):
        h = self.lift(x)
        for layer in self.layers:
            h = layer(h, src, dst, edge_sh, edge_dist_norm)
        return self.classifier(self.to_scalars(h))


model = SE3GNN(irreps_in, irreps_sh, HIDDEN_MUL, N_LAYERS, n_classes=2).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
hidden_str = f"{HIDDEN_MUL}x0e + {HIDDEN_MUL//2}x1o + {HIDDEN_MUL//4}x2e"
print(f"\nModel: {n_params:,} parameters")
print(f"  Hidden irreps : {hidden_str}")
print(f"  Layers        : {N_LAYERS}   Lmax: {LMAX}")


# ── 5. Training ───────────────────────────────────────────────────────────────
class_weights = torch.tensor([1.0, 3.0]).to(device)
print(f"\nClass weights — ground: {class_weights[0]:.2f}  wall: {class_weights[1]:.2f}")

CKPT_PATH = "data/processed/se3_model.pt"

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

node_feat_d      = node_feat.to(device)
src_d            = src.to(device)
dst_d            = dst.to(device)
edge_sh_d        = edge_sh.to(device)
edge_dist_norm_d = edge_dist_norm.to(device)
labels_d         = torch.tensor(labels_np).to(device)

import os
if os.path.exists(CKPT_PATH):
    print(f"\nCheckpoint found — loading {CKPT_PATH} (skipping training)")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
else:
    print(f"\nTraining for {N_EPOCHS} epochs ...")
    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(node_feat_d, src_d, dst_d, edge_sh_d, edge_dist_norm_d)
        loss   = F.cross_entropy(logits, labels_d, weight=class_weights)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds    = logits.argmax(dim=-1)
                acc      = (preds == labels_d).float().mean().item()
                tp       = ((preds == 1) & (labels_d == 1)).sum().item()
                wall_rec = tp / max((labels_d == 1).sum().item(), 1)
                wall_pre = tp / max((preds == 1).sum().item(), 1)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.3f}  "
                  f"wall precision={wall_pre:.3f}  recall={wall_rec:.3f}  "
                  f"[{elapsed:.0f}s elapsed]")

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"  Model saved → {CKPT_PATH}")


# ── 6. Inference ──────────────────────────────────────────────────────────────
print("\nRunning inference on all points ...")
model.eval()
with torch.no_grad():
    logits = model(node_feat_d, src_d, dst_d, edge_sh_d, edge_dist_norm_d)
    probs  = F.softmax(logits, dim=-1)
    preds  = logits.argmax(dim=-1)

pred_np = preds.cpu().numpy().astype(np.uint8)
prob_np = probs[:, 1].cpu().numpy().astype(np.float32)
print(f"  Predicted wall: {pred_np.sum():,} / {N:,} points")


# ── 7. Save predictions to LAZ ────────────────────────────────────────────────
print(f"\nSaving → {OUT_PATH}")
out = laspy.LasData(header=las.header)
out.points = las.points.copy()

out.add_extra_dims([
    laspy.ExtraBytesParams(name="wall_prob",  type=np.float32,
                           description="SE(3) GNN wall probability"),
    laspy.ExtraBytesParams(name="pred_class", type=np.uint8,
                           description="0=ground 1=wall"),
])
out.wall_prob  = prob_np
out.pred_class = pred_np
out.write(OUT_PATH)
print("Saved.")


# ── 8. Visualise ──────────────────────────────────────────────────────────────
x_np = pos_np[:, 0] - pos_np[:, 0].mean()
y_np = pos_np[:, 1] - pos_np[:, 1].mean()
z_np = pos_np[:, 2]
pts  = np.column_stack([x_np, y_np, z_np])

cloud = pv.PolyData(pts)
cloud["Wall Probability"] = prob_np
cloud["Predicted Class"]  = pred_np.astype(float)
cloud["Pseudo-label"]     = labels_np.astype(float)

pl = pv.Plotter(shape=(1, 3), window_size=(2100, 750))

pl.subplot(0, 0)
pl.set_background("black")
pl.add_text("Pseudo-labels (curvature)", font_size=9, color="white")
pl.add_points(cloud, scalars="Pseudo-label", cmap="coolwarm",
              clim=[0, 1], point_size=3, render_points_as_spheres=False)
pl.add_scalar_bar("0=ground  1=wall", color="white", fmt="%.0f")
pl.show_axes()

pl.subplot(0, 1)
pl.set_background("black")
pl.add_text("SE(3) GNN — Wall Probability", font_size=9, color="white")
pl.add_points(cloud, scalars="Wall Probability", cmap="inferno",
              clim=[0, 1], point_size=3, render_points_as_spheres=False)
pl.add_scalar_bar("P(wall)", color="white", fmt="%.2f")
pl.show_axes()

pl.subplot(0, 2)
pl.set_background("black")
pl.add_text("SE(3) GNN — Predicted Class", font_size=9, color="white")
pl.add_points(cloud, scalars="Predicted Class", cmap="coolwarm",
              clim=[0, 1], point_size=3, render_points_as_spheres=False)
pl.add_scalar_bar("0=ground  1=wall", color="white", fmt="%.0f")
pl.show_axes()

pl.link_views()
pl.show()
