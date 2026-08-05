#!/usr/bin/env python3
"""
mesh_demo.py  --  2D illustration of the data-driven Voronoi mesher.

A teaching/diagnostic toy that mirrors ``_iterator._generate_voronoi_cells``
in two dimensions (a vertical slice: x = horizontal km, z = depth km):

    1. coarse INITIAL mesh, seed density biased to the data (ray) density
    2. assign every ray-path point to its nearest seed (a Voronoi cell)
    3. SPLIT any cell with > target_rays rays that is still wider than
       min_cell_width, for several passes
    4. CULL empty cells
    5. BACKFILL empty volume with COARSE filler keyed to max_cell_width

ALL distances are taken in a metric space where the vertical (depth) axis
is scaled by ``hvr`` (horizontal-to-vertical ratio), exactly as the real
code stretches the radial axis inside _to_xyz.  hvr > 1 makes cells
vertically thin -- the "Nx vert compression" your log reports.

Straight rays, Euclidean, no velocity model: the point is the *meshing*.

Usage
-----
    python mesh_demo.py
    python mesh_demo.py --events 300 --stations 25 --hvr 3
    python mesh_demo.py --hvr 1 --no-backfill --clusters 1 --seed 7
"""

import argparse
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CONFIG = dict(
    width=400.0, depth=200.0,                 # domain (km)
    n_events=250, n_stations=20, clusters=3,  # data
    max_rays=4000,
    max_cell_width=70.0, min_cell_width=12.0, # meshing knobs (km)
    target_rays=30, min_rays=10,
    max_passes=10, max_seeds=2000, enable_backfill=True,
    hvr=2.0,                                  # horizontal-to-vertical ratio
    seed=1, out="mesh_demo.png",
)


# --------------------------------------------------------------------------
# Metric transform: scale depth by hvr so all NN/Voronoi distances treat
# vertical separation as hvr-times larger (=> vertically compressed cells).
# --------------------------------------------------------------------------
def _metric(pts, hvr):
    out = np.asarray(pts, dtype=float).copy()
    out[..., 1] *= hvr
    return out


def _unmetric(pts, hvr):
    """Inverse of _metric: undo the depth stretch (xyz -> real coords)."""
    out = np.asarray(pts, dtype=float).copy()
    out[..., 1] /= hvr
    return out


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def make_data(cfg, rng):
    W, D = cfg["width"], cfg["depth"]
    sx = rng.uniform(0.05 * W, 0.95 * W, cfg["n_stations"])
    stations = np.column_stack([sx, np.zeros_like(sx)])

    ne = cfg["n_events"]
    if cfg["clusters"] <= 0:
        ex = rng.uniform(0.05 * W, 0.95 * W, ne)
        ez = rng.uniform(0.15 * D, 0.95 * D, ne)
    else:
        cx = rng.uniform(0.15 * W, 0.85 * W, cfg["clusters"])
        cz = rng.uniform(0.20 * D, 0.85 * D, cfg["clusters"])
        spread = 0.06 * W
        which = rng.integers(0, cfg["clusters"], ne)
        ex = np.clip(cx[which] + rng.normal(0, spread, ne), 1, W - 1)
        ez = np.clip(cz[which] + rng.normal(0, spread, ne), 1, D - 1)
    return np.column_stack([ex, ez]), stations


def make_rays(events, stations, cfg, rng):
    pairs = [(e, s) for e in events for s in stations]
    if len(pairs) > cfg["max_rays"]:
        idx = rng.choice(len(pairs), cfg["max_rays"], replace=False)
        pairs = [pairs[i] for i in idx]
    pts, ids, segs = [], [], []
    for rid, (e, s) in enumerate(pairs):
        n = max(8, int(np.hypot(*(e - s)) / 3.0))
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(s[None, :] * (1 - t) + e[None, :] * t)
        ids.append(np.full(n, rid))
        segs.append(np.vstack([s, e]))
    return np.vstack(pts), np.concatenate(ids), segs


# --------------------------------------------------------------------------
# Mesh construction (mirrors _generate_voronoi_cells)
# --------------------------------------------------------------------------
def initial_seeds(ray_pts, cfg, rng):
    """Coarse, density-biased initial seeds.

    n_initial ~ metric_area / max_cell_width^2 (2D analogue of
    metric_volume / width^3); metric_area includes the hvr stretch, so a
    larger hvr asks for more (vertically thinner) cells -- just like the
    real code derives n_initial from _to_xyz(corners).
    """
    W, D, hvr = cfg["width"], cfg["depth"], cfg["hvr"]
    metric_area = W * (D * hvr)
    n_initial = int(np.clip(metric_area / cfg["max_cell_width"] ** 2, 8, 600))

    nb = max(4, int(round(np.sqrt(n_initial))) * 2)
    H, xe, ze = np.histogram2d(ray_pts[:, 0], ray_pts[:, 1],
                               bins=nb, range=[[0, W], [0, D]])
    dens = H.ravel().astype(float)
    dens = dens / dens.sum() if dens.sum() > 0 else np.ones_like(dens)
    prob = 0.7 * dens + 0.3 * np.ones_like(dens) / dens.size
    prob /= prob.sum()

    idx = rng.choice(dens.size, size=n_initial, p=prob)
    ix, iz = np.unravel_index(idx, (nb, nb))
    sx = xe[ix] + rng.random(n_initial) * (xe[1] - xe[0])
    sz = ze[iz] + rng.random(n_initial) * (ze[1] - ze[0])
    return np.column_stack([np.clip(sx, 0, W), np.clip(sz, 0, D)]), n_initial


def _cell_widths(seeds, hvr):
    """NN distance per seed IN METRIC SPACE -> cell-width proxy."""
    if len(seeds) < 2:
        return np.full(len(seeds), np.inf)
    d, _ = cKDTree(_metric(seeds, hvr)).query(_metric(seeds, hvr), k=2)
    return d[:, 1]


def refine(seeds, ray_pts, ray_id, cfg, rng):
    """Top-down refinement, mirroring _generate_voronoi_cells.

    Each pass (in the hvr-scaled metric space):
      - assign every ray point to its nearest seed (Voronoi)
      - count UNIQUE rays per cell; estimate cell width = NN distance
      - split any cell over target_rays whose children would still clear
        min_cell_width, using the source's estimate r_cell / 1.587
      - a split REPLACES the parent with 3-5 children scattered
        ISOTROPICALLY in metric space at scale (nn*0.5)/2, then mapped back
        to real coords.  The inverse metric divides depth by hvr, so an
        isotropic metric blob lands as a vertically-squashed cluster -- the
        compression is baked into the subdivision, not fitted to the data.
    Loops to convergence (no cell needs splitting) up to max_passes.
    """
    target, min_w, hvr = cfg["target_rays"], cfg["min_cell_width"], cfg["hvr"]
    rp_metric = _metric(ray_pts, hvr)
    n_splits, passes_used = 0, 0

    for _ in range(cfg["max_passes"]):
        passes_used += 1
        seeds_metric = _metric(seeds, hvr)
        tree = cKDTree(seeds_metric)
        if len(seeds) > 1:
            nn = tree.query(seeds_metric, k=2)[0][:, 1]
        else:
            nn = np.full(len(seeds), np.inf)
        _, nearest = tree.query(rp_metric)

        sets = [set() for _ in range(len(seeds))]
        for c, rid in zip(nearest, ray_id):
            sets[c].add(rid)
        counts = np.array([len(s) for s in sets])

        # 1.587 = 4**(1/3): the source's estimate of child width after a
        # 4-way split, kept verbatim so the rule matches the code you'll
        # read (a strict 2D analogue would be 2.0).
        to_split = np.array([i for i in range(len(seeds))
                             if counts[i] > target
                             and (min_w == 0 or nn[i] / 1.587 >= min_w)])
        if to_split.size == 0 or len(seeds) >= cfg["max_seeds"]:
            break
        rng.shuffle(to_split)                          # a touch of chaos

        keep = np.ones(len(seeds), dtype=bool)
        children = []
        for i in to_split:
            keep[i] = False
            parent_m = seeds_metric[i]
            r_split = nn[i] * 0.5
            n_child = int(rng.choice([3, 4, 5]))
            off = rng.normal(scale=r_split / 2.0, size=(n_child, 2))
            kids = _unmetric(parent_m + off, hvr)      # back to real coords
            kids[:, 0] = np.clip(kids[:, 0], 0, cfg["width"])
            kids[:, 1] = np.clip(kids[:, 1], 0, cfg["depth"])
            children.append(kids)
        n_splits += len(to_split)                      # count parents split
        seeds = np.vstack([seeds[keep]] + children)

    return seeds, n_splits, passes_used


def cull_empty(seeds, ray_pts, cfg):
    _, nearest = cKDTree(_metric(seeds, cfg["hvr"])).query(_metric(ray_pts, cfg["hvr"]))
    keep = np.zeros(len(seeds), dtype=bool)
    keep[np.unique(nearest)] = True
    return seeds[keep], int((~keep).sum())


def backfill(data_seeds, cfg, rng):
    """COARSE filler keyed to max_cell_width, in metric space."""
    if not cfg["enable_backfill"] or len(data_seeds) == 0:
        return np.empty((0, 2))
    W, D, hvr = cfg["width"], cfg["depth"], cfg["hvr"]
    spacing = cfg["max_cell_width"]
    n_cand = int(np.clip(W * (D * hvr) / spacing ** 2 * 2.0, 16, 1000))
    cand = np.column_stack([rng.uniform(0, W, n_cand), rng.uniform(0, D, n_cand)])
    d, _ = cKDTree(_metric(data_seeds, hvr)).query(_metric(cand, hvr))
    return cand[d > spacing]


# --------------------------------------------------------------------------
# Diagnostics (mirror the log)
# --------------------------------------------------------------------------
def diagnose(tag, all_seeds, n_data, ray_pts, ray_id, cfg,
             n_initial, n_splits, passes_used, n_culled):
    hvr = cfg["hvr"]
    n_total = len(all_seeds)
    n_back = n_total - n_data

    _, nearest = cKDTree(_metric(all_seeds, hvr)).query(_metric(ray_pts, hvr))
    sets = [set() for _ in range(n_total)]
    for c, rid in zip(nearest, ray_id):
        sets[c].add(rid)
    rpc = np.array([len(s) for s in sets])
    pop = rpc[rpc > 0]

    dw = _cell_widths(all_seeds[:n_data], hvr)
    dw = dw[np.isfinite(dw)]

    print(f"  [{tag}] Adaptive mesh: {n_total} cells "
          f"({n_data} data-driven, {n_back} backfill); started with {n_initial}, "
          f"performed {n_splits} splits in {passes_used} passes, "
          f"culled {n_culled} empty cells")
    if dw.size:
        p = np.percentile(dw, [10, 25, 75, 90])
        print(f"  [{tag}] Cell width (km, {hvr:.1f}x vert compression, data-driven only): "
              f"10%: {p[0]:.1f} 25%: {p[1]:.1f} 75%: {p[2]:.1f} 90%: {p[3]:.1f}, "
              f"mean/median = {dw.mean():.1f}/{np.median(dw):.1f}")
    cov = len(pop)
    if cov:
        print(f"  [{tag}] Ray coverage: {cov}/{n_total} cells ({100*cov/n_total:.1f}%), "
              f"avg rays-per-sampled-cell: {pop.mean():.1f}")
        under = int((pop < cfg["min_rays"]).sum())
        well = int((pop >= cfg["target_rays"]).sum())
        q = np.percentile(pop, [10, 25, 50, 75, 90])
        print(f"  [{tag}]  Mesh diagnostic (target {cfg['target_rays']} rays/cell):")
        print(f"  [{tag}]    {cov} populated cells: {under} ({100*under/cov:.1f}%) "
              f"under min_rays ({cfg['min_rays']}), {well} ({100*well/cov:.1f}%) well-resolved")
        print(f"  [{tag}]    rays-per-cell 10%: {q[0]:.0f}, 25%: {q[1]:.0f}, "
              f"50%: {q[2]:.0f}, 75%: {q[3]:.0f}, 90%: {q[4]:.0f}")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def raster_cells(ax, seeds, n_data, ray_pts, ray_id, cfg, res=320):
    """Rasterise the (metric) Voronoi diagram and colour each cell by fate:

        backfill cell (kept filler)        -> grey
        data cell, 0 rays (will be culled) -> white
        data cell, 0 < rays < min_rays     -> faded colour (under-resolved)
        data cell, rays >= min_rays        -> full colour
    """
    W, D, hvr = cfg["width"], cfg["depth"], cfg["hvr"]
    min_rays = cfg["min_rays"]
    nx = res
    nz = max(8, int(res * D / W))
    gx, gz = np.meshgrid(np.linspace(0, W, nx), np.linspace(0, D, nz))
    grid = np.column_stack([gx.ravel(), gz.ravel()])
    tree = cKDTree(_metric(seeds, hvr))
    _, nearest = tree.query(_metric(grid, hvr))
    nearest = nearest.reshape(nz, nx)

    # unique-ray count per cell, against THIS seed set so the colours match
    # the tessellation actually drawn
    _, owner = tree.query(_metric(ray_pts, hvr))
    sets = [set() for _ in range(len(seeds))]
    for c, rid in zip(owner, ray_id):
        sets[c].add(rid)
    counts = np.array([len(s) for s in sets])

    cmap = plt.get_cmap("tab20")
    white = np.array([1.0, 1.0, 1.0])
    under_alpha = 0.35                       # opacity of under-resolved cells
    rgb = np.ones((nz, nx, 3))               # default white (covers culled)

    for c in range(len(seeds)):
        m = nearest == c
        if not m.any():
            continue
        if c >= n_data:                      # backfill filler (kept)
            rgb[m] = 0.86 if (c % 2 == 0) else 0.78
        elif counts[c] == 0:                 # empty data cell -> culled
            rgb[m] = white
        else:
            col = np.array(cmap((c % 20) / 20.0)[:3])
            if counts[c] < min_rays:         # under-resolved -> fade to bg
                col = under_alpha * col + (1.0 - under_alpha) * white
            rgb[m] = col

    edge = np.zeros((nz, nx), bool)
    edge[:, :-1] |= nearest[:, :-1] != nearest[:, 1:]
    edge[:-1, :] |= nearest[:-1, :] != nearest[1:, :]
    rgb[edge] = 0.15
    ax.imshow(rgb, origin="upper", extent=[0, W, D, 0], aspect="auto",
              interpolation="nearest")


def plot(events, stations, segs, ray_pts, ray_id, init_seeds, final_seeds, n_data, cfg):
    W, D = cfg["width"], cfg["depth"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    for ax, seeds, ndat, title in (
        (axes[0], init_seeds, len(init_seeds), "Initial mesh (coarse, density-biased)"),
        (axes[1], final_seeds, n_data, "Final mesh (refined + coarse backfill)"),
    ):
        raster_cells(ax, seeds, ndat, ray_pts, ray_id, cfg)
        step = max(1, len(segs) // 220)
        for seg in segs[::step]:
            ax.plot(seg[:, 0], seg[:, 1], color="k", lw=0.25, alpha=0.18)
        ax.scatter(seeds[:ndat, 0], seeds[:ndat, 1], s=10, c="white",
                   edgecolors="k", linewidths=0.5, zorder=5)
        if len(seeds) > ndat:
            ax.scatter(seeds[ndat:, 0], seeds[ndat:, 1], s=10, c="0.4",
                       marker="x", linewidths=0.7, zorder=5)
        ax.scatter(events[:, 0], events[:, 1], s=14, c="red", marker="*",
                   edgecolors="k", linewidths=0.3, zorder=6)
        ax.scatter(stations[:, 0], stations[:, 1], s=275, c="deepskyblue",
                   marker="v", edgecolors="k", linewidths=0.7, zorder=7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (km)"); ax.set_ylabel("depth (km)")
        ax.set_xlim(0, W); ax.set_ylim(D, 0)

    legend = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="k", markersize=10, label="events"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="deepskyblue",
               markeredgecolor="k", markersize=14, label="stations"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="k", markersize=8, label="data-driven cell seed"),
        Line2D([0], [0], marker="x", color="0.4", linestyle="None",
               markersize=8, label="backfill cell seed"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="0.83",
               markeredgecolor="k", markersize=9, label="under min_rays (faded) / culled (white)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)
    fig.suptitle(
        f"2D Voronoi mesh demo  |  {cfg['n_events']} events, {cfg['n_stations']} stations  |  "
        f"target {cfg['target_rays']} rays/cell, cell width {cfg['min_cell_width']:.0f}-"
        f"{cfg['max_cell_width']:.0f} km  |  hvr {cfg['hvr']:.1f} (vert compression)  |  "
        f"backfill {'ON' if cfg['enable_backfill'] else 'OFF'}", fontsize=12)
    fig.savefig(cfg["out"], dpi=130, bbox_inches="tight")
    print(f"\nFigure written to {cfg['out']}")


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="2D Voronoi mesh demo")
    p.add_argument("--events", type=int, dest="n_events")
    p.add_argument("--stations", type=int, dest="n_stations")
    p.add_argument("--clusters", type=int)
    p.add_argument("--target-rays", type=int, dest="target_rays")
    p.add_argument("--min-rays", type=int, dest="min_rays")
    p.add_argument("--min-cell-width", type=float, dest="min_cell_width")
    p.add_argument("--max-cell-width", type=float, dest="max_cell_width")
    p.add_argument("--hvr", type=float)
    p.add_argument("--passes", type=int, dest="max_passes")
    p.add_argument("--no-backfill", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--out", type=str)
    return p.parse_args()


def main():
    cfg = dict(CONFIG)
    args = parse_args()
    for k, v in vars(args).items():
        if k == "no_backfill":
            continue
        if v is not None:
            cfg[k] = v
    if args.no_backfill:
        cfg["enable_backfill"] = False

    rng = np.random.default_rng(cfg["seed"])
    events, stations = make_data(cfg, rng)
    ray_pts, ray_id, segs = make_rays(events, stations, cfg, rng)
    print(f"Built {len(np.unique(ray_id))} rays "
          f"({len(events)} events x {len(stations)} stations), "
          f"{len(ray_pts)} path points  |  hvr = {cfg['hvr']:.1f}\n")

    seeds, n_initial = initial_seeds(ray_pts, cfg, rng)
    init_seeds = seeds.copy()
    diagnose("INITIAL", seeds, len(seeds), ray_pts, ray_id, cfg, n_initial, 0, 0, 0)

    seeds, n_splits, passes_used = refine(seeds, ray_pts, ray_id, cfg, rng)
    seeds, n_culled = cull_empty(seeds, ray_pts, cfg)
    n_data = len(seeds)

    filler = backfill(seeds, cfg, rng)
    final_seeds = np.vstack([seeds, filler]) if len(filler) else seeds

    print()
    diagnose("FINAL", final_seeds, n_data, ray_pts, ray_id, cfg,
             n_initial, n_splits, passes_used, n_culled)

    plot(events, stations, segs, ray_pts, ray_id, init_seeds, final_seeds, n_data, cfg)


if __name__ == "__main__":
    main()