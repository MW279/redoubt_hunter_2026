"""
SE(3)-Equivariant Transformer GNN for archaeological earthwork detection.

Upgrades over train_se3.py:
  1. SE3TransformerLayer: attention-weighted message passing
       - MLP([scalar_i, scalar_j, dist]) -> sigmoid gate multiplied onto messages
  2. MultiScaleSE3GNN: local (0.8m) + macro (4.0m) dual-stream graph
       - Parallel layer stacks, concatenated scalar readout before classifier
"""

import os
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
OUT_PATH      = "data/processed/predictions_02_transformer.laz"
CKPT_PATH     = "data/processed/se3_transformer_model.pt"

LOCAL_RADIUS  = 0.8    # local graph radius in metres
MACRO_RADIUS  = 4.0    # macro graph radius in metres
MAX_NEIGHBORS = 16
LMAX          = 2
HIDDEN_MUL    = 8
N_LAYERS      = 3
N_EPOCHS      = 60
LR            = 3e-3
EDGE_BATCH    = 500_000
CURVATURE_PCT = 93

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
threshold = float(np.percentile(curv_np, CURVATURE_PCT))
labels_np = (curv_np > threshold).astype(np.int64)
n_wall    = labels_np.sum()
print(f"  Curvature threshold (p{CURVATURE_PCT}): {threshold:.6f}")
print(f"  Pseudo-labels — ground: {N - n_wall:,}  wall: {n_wall:,}")


# ── 2. Build two radius graphs ─────────────────────────────────────────────────
def build_graph(pos_np, radius, max_neighbors, irreps_sh):
    print(f"  Building graph (r={radius}m, max {max_neighbors} neighbours) ...")
    tree = KDTree(pos_np)
    # k-NN query returns a compact numpy array instead of a Python list-of-lists,
    # avoiding the ~1 GB of Python object overhead from query_ball_point at large radii.
    dists, idxs = tree.query(pos_np, k=max_neighbors + 1, workers=-1)
    # Column 0 is always the self-match at distance 0; drop it.
    dists = dists[:, 1:]   # [N, max_neighbors]
    idxs  = idxs[:, 1:]    # [N, max_neighbors]

    # Keep only edges within the radius (vectorised, no Python loop)
    valid   = dists <= radius                                     # [N, max_neighbors] bool
    src_arr = np.repeat(np.arange(len(pos_np)), max_neighbors)[valid.ravel()]
    dst_arr = idxs.ravel()[valid.ravel()]

    src = torch.tensor(src_arr, dtype=torch.long)
    dst = torch.tensor(dst_arr, dtype=torch.long)

    pos_t          = torch.tensor(pos_np)
    edge_vec       = pos_t[dst] - pos_t[src]
    edge_dist_norm = edge_vec.norm(dim=-1, keepdim=True) / radius  # normalise to [0,1]
    edge_sh        = o3.spherical_harmonics(
        irreps_sh, edge_vec, normalize=True, normalization="component"
    )

    print(f"    {len(src):,} edges  ({len(src) / len(pos_np):.1f} per node)")
    return {"src": src, "dst": dst, "edge_sh": edge_sh, "edge_dist_norm": edge_dist_norm}


irreps_sh = o3.Irreps.spherical_harmonics(lmax=LMAX)  # "1x0e + 1x1o + 1x2e"

print("\nBuilding radius graphs ...")
local_graph = build_graph(pos_np, LOCAL_RADIUS,  MAX_NEIGHBORS, irreps_sh)
macro_graph = build_graph(pos_np, MACRO_RADIUS, MAX_NEIGHBORS, irreps_sh)


# ── 3. Node feature tensor ─────────────────────────────────────────────────────
# Layout: irreps_in = "1x0e + 1x1o"  →  [curvature, nx, ny, nz]
node_feat = torch.cat([
    torch.tensor(curv_np).unsqueeze(-1),
    torch.tensor(normal_np),
], dim=-1)  # [N, 4]

irreps_in = o3.Irreps("1x0e + 1x1o")
assert node_feat.shape[-1] == irreps_in.dim


# ── 4. Model ──────────────────────────────────────────────────────────────────
class SE3TransformerLayer(nn.Module):
    """
    SE(3)-equivariant message passing with learned attention gate.

    Standard radial-weighted tensor product messages are scaled by a
    sigmoid attention score computed from invariant (0e) scalars of the
    sender/receiver nodes and the normalised edge distance.
    """
    def __init__(self, irreps_in, irreps_sh, irreps_out):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in, irreps_sh, irreps_out, shared_weights=False,
        )
        self.radial = nn.Sequential(
            nn.Linear(1, 32),  nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, self.tp.weight_numel),
        )
        self.skip = o3.Linear(irreps_in, irreps_out)

        # Extract invariant scalars for the attention gate
        n_scalars = sum(mul for mul, ir in irreps_in if ir == o3.Irrep("0e"))
        self.to_scalars = o3.Linear(irreps_in, o3.Irreps(f"{n_scalars}x0e"))
        self.attn_mlp = nn.Sequential(
            nn.Linear(n_scalars * 2 + 1, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x, src, dst, edge_sh, edge_dist_norm, edge_batch=500_000):
        E       = src.shape[0]
        out_dim = self.tp.irreps_out.dim
        scalars = self.to_scalars(x)  # [N, n_scalars]

        chunks = []
        for s in range(0, E, edge_batch):
            e = min(s + edge_batch, E)
            w       = self.radial(edge_dist_norm[s:e])
            m       = self.tp(x[src[s:e]], edge_sh[s:e], w)
            attn_in = torch.cat(
                [scalars[src[s:e]], scalars[dst[s:e]], edge_dist_norm[s:e]], dim=-1
            )
            alpha   = self.attn_mlp(attn_in).clamp(min=0.1, max=1.0)  # epsilon floor prevents gate collapse
            chunks.append(m * alpha)

        messages = torch.cat(chunks, dim=0)  # [E, out_dim]

        # Mean aggregation (equivariant — linear in messages)
        idx = dst.unsqueeze(-1).expand_as(messages)
        agg = torch.zeros(x.shape[0], out_dim, device=x.device).scatter_add(0, idx, messages)
        cnt = torch.zeros(x.shape[0], 1, device=x.device).scatter_add(
            0, dst.unsqueeze(-1), torch.ones(E, 1, device=x.device)
        )
        agg = agg / cnt.clamp(min=1)

        return agg + self.skip(x)


class MultiScaleSE3GNN(nn.Module):
    """
    Dual-stream SE(3)-equivariant GNN.

    Lifted features are passed independently through a local-graph stream
    and a macro-graph stream.  The invariant scalar outputs of both streams
    are concatenated before the final classifier.
    """
    def __init__(self, irreps_in, irreps_sh, hidden_mul, n_layers, n_classes):
        super().__init__()
        irreps_h = o3.Irreps(
            f"{hidden_mul}x0e + {hidden_mul // 2}x1o + {hidden_mul // 4}x2e"
        )
        self.lift = o3.Linear(irreps_in, irreps_h)

        self.local_layers = nn.ModuleList([
            SE3TransformerLayer(irreps_h, irreps_sh, irreps_h) for _ in range(n_layers)
        ])
        self.macro_layers = nn.ModuleList([
            SE3TransformerLayer(irreps_h, irreps_sh, irreps_h) for _ in range(n_layers)
        ])

        n_scalars = hidden_mul
        self.to_scalars_local = o3.Linear(irreps_h, o3.Irreps(f"{n_scalars}x0e"))
        self.to_scalars_macro = o3.Linear(irreps_h, o3.Irreps(f"{n_scalars}x0e"))
        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_scalars * 2, 32),
            nn.SiLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, local_graph, macro_graph):
        h = self.lift(x)

        h_local = h
        for layer in self.local_layers:
            h_local = layer(h_local, **local_graph)

        h_macro = h
        for layer in self.macro_layers:
            h_macro = layer(h_macro, **macro_graph)

        s_local = self.to_scalars_local(h_local)
        s_macro = self.to_scalars_macro(h_macro)
        return self.classifier(torch.cat([s_local, s_macro], dim=-1))


model = MultiScaleSE3GNN(irreps_in, irreps_sh, HIDDEN_MUL, N_LAYERS, n_classes=2).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
hidden_str = f"{HIDDEN_MUL}x0e + {HIDDEN_MUL//2}x1o + {HIDDEN_MUL//4}x2e"
print(f"\nModel: {n_params:,} parameters")
print(f"  Hidden irreps : {hidden_str}")
print(f"  Layers        : {N_LAYERS} local + {N_LAYERS} macro   Lmax: {LMAX}")
print(f"  Local radius  : {LOCAL_RADIUS}m    Macro radius: {MACRO_RADIUS}m")


# ── 5. Training ───────────────────────────────────────────────────────────────
class_weights = torch.tensor([1.0, 3.0]).to(device)
print(f"\nClass weights — ground: {class_weights[0]:.2f}  wall: {class_weights[1]:.2f}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

node_feat_d   = node_feat.to(device)
labels_d      = torch.tensor(labels_np).to(device)
local_graph_d = {k: v.to(device) for k, v in local_graph.items()}
macro_graph_d = {k: v.to(device) for k, v in macro_graph.items()}

if os.path.exists(CKPT_PATH):
    print(f"\nCheckpoint found — loading {CKPT_PATH} (skipping training)")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
else:
    print(f"\nTraining for {N_EPOCHS} epochs ...")
    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(node_feat_d, local_graph_d, macro_graph_d)
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
    logits = model(node_feat_d, local_graph_d, macro_graph_d)
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
                           description="SE3 Transformer wall probability"),
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
pl.add_text("SE(3) Transformer — Wall Probability", font_size=9, color="white")
pl.add_points(cloud, scalars="Wall Probability", cmap="inferno",
              clim=[0, 1], point_size=3, render_points_as_spheres=False)
pl.add_scalar_bar("P(wall)", color="white", fmt="%.2f")
pl.show_axes()

pl.subplot(0, 2)
pl.set_background("black")
pl.add_text("SE(3) Transformer — Predicted Class", font_size=9, color="white")
pl.add_points(cloud, scalars="Predicted Class", cmap="coolwarm",
              clim=[0, 1], point_size=3, render_points_as_spheres=False)
pl.add_scalar_bar("0=ground  1=wall", color="white", fmt="%.0f")
pl.show_axes()

pl.link_views()
pl.show()
