"""
make_movie.py - run one simulation without a window and save the results.
=========================================================================

    python make_movie.py                        # defaults (see simulation.Params)
    python make_movie.py --pattern voronoi --seed 7 --cells 500 --steps 3000

Writes to --out (default: ./output):
    basalt_evolution.gif   animation (surface cracking, then the cracks growing down)
    basalt_cracking.png    the surface cracking at 4 moments ("growth" start only)
    basalt_snapshots.png   network + angle histogram at 4 depths
    basalt_summary.png     6 measured curves vs depth
"""

import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402

from patterns import PATTERNS                      # noqa: E402
from plots import (Dashboard, plot_snapshots, plot_summary, plot_formation,  # noqa: E402
                   save_gif, fig_to_image)
from simulation import Simulation, Params          # noqa: E402


def make_movie(p: Params, out=".", n_frames=110, gif=True):
    os.makedirs(out, exist_ok=True)
    sim = Simulation(p)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=90) if gif else None
    frames = []

    # ---- phase 1: the surface cracks (only for the "growth" start)
    if sim.forming:
        print(f"surface cracking: up to {sim.growth.N} cracks grow from intact crust ...")
        dash = Dashboard(fig, sim, rect=(0.02, 0.04, 0.97, 0.93)) if gif else None
        form_snaps = []
        while sim.forming:
            g = sim.growth
            if g.step_no in (8, 20, 32) or (g.done and not form_snaps[-1:] == [g.step_no]):
                form_snaps.append((g.step_no, copy.deepcopy(g)))
            if gif:
                dash.update()
                frames.append(fig_to_image(fig))
            if g.done:
                form_snaps.append((g.step_no, copy.deepcopy(g)))
            sim.advance(1)
        g = sim.growth
        if form_snaps[-1][0] != g.step_no:
            form_snaps.append((g.step_no, copy.deepcopy(g)))
        ma = np.array(g.meet_angles)
        print(f"  {len(g.cracks)} cracks, {g.n_junctions} T junctions after {g.step_no} steps; "
              f"{(ma > 75).mean():.0%} of cracks met an older one within 15° of a right angle")
        plot_formation(os.path.join(out, "basalt_cracking.png"), form_snaps[-4:])
        if gif:
            plt.close(fig)
            fig = plt.figure(figsize=(12.8, 7.2), dpi=90)

    # ---- phase 2: the cracks grow downward
    m0 = sim.history[0]
    print(f"start of the depth phase: {sim.net.ncell} columns, "
          f"{m0['frac_T']:.0%} T-junctions, {m0['frac'][6]:.0%} hexagons")
    print(f"gradient check (analytic vs finite difference): max rel. error "
          f"{sim.gradient_error():.1e}")
    snap_steps = sorted({0, min(10, p.steps), min(60, p.steps), p.steps})
    frame_steps = sorted({0, p.steps} | set(snap_steps)
                         | set(np.round(np.geomspace(1, p.steps, n_frames)).astype(int)))
    snaps = []
    dash = Dashboard(fig, sim, rect=(0.02, 0.04, 0.97, 0.93)) if gif else None
    for k in frame_steps:
        sim.advance(k - sim.step_no, record_every=1 if k < 100 else 10)
        if k in snap_steps:
            snaps.append((k, copy.deepcopy(sim.net)))
        if gif:
            dash.update()
            frames.append(fig_to_image(fig))
        if k in snap_steps:
            h = sim.history[-1]
            print(f"  step {k:5d}: largest junction angle {h['largest_angle']:6.1f}°  "
                  f"hexagons {h['frac'][6]:5.1%}  J/A² {h['shape_J']:.4f}  "
                  f"P/√A {h['shape_P']:.3f}")

    if gif:
        save_gif(os.path.join(out, "basalt_evolution.gif"), frames)
        plt.close(fig)
    plot_snapshots(os.path.join(out, "basalt_snapshots.png"), snaps)
    plot_summary(os.path.join(out, "basalt_summary.png"), sim.history)
    print("wrote results to", os.path.abspath(out))
    return sim


if __name__ == "__main__":
    d = Params()
    ap = argparse.ArgumentParser(description="Basalt column simulation -> GIF + figures")
    ap.add_argument("--pattern", default=d.pattern, choices=list(PATTERNS))
    ap.add_argument("--cells", type=int, default=d.n_cells)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--gamma", type=float, default=d.gamma, help="crack-energy weight")
    ap.add_argument("--dt", type=float, default=d.dt, help="step size of the gradient flow")
    ap.add_argument("--steps", type=int, default=d.steps)
    ap.add_argument("--frames", type=int, default=110)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    make_movie(Params(pattern=a.pattern, n_cells=a.cells, seed=a.seed, gamma=a.gamma,
                      dt=a.dt, steps=a.steps), a.out, a.frames, not a.no_gif)
