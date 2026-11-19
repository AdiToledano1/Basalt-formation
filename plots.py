"""
plots.py - drawing.  No physics here.
=====================================

Dashboard      the live 4-panel view (network + 3 charts), used both by the
               control panel and by the movie maker
plot_summary   6 curves vs depth, with square / hexagon reference lines
plot_snapshots the network and its angle histogram at a few depths
save_gif       write frames to an animated GIF
"""

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from physics import L, unwrap, REFERENCE
from measure import measure, corner_angles, T_LIKE, CORNER_CUT

SIDE_COLORS = {3: "#e34948", 4: "#eb6834", 5: "#eda100", 6: "#2a78d6",
               7: "#6250d6", 8: "#e87ba4"}
INK, INK2, GRID = "#222222", "#6b6a66", "#e4e3df"
ANGLE_BINS = np.arange(60, 201, 4)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def log_depth_axis(ax, steps):
    ax.set_xscale("log")
    ax.set_xlim(0.7, max(steps, 10) * 1.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def depth_x(history):
    """Step 0 cannot sit on a log axis; draw it at 0.7."""
    return np.maximum([h["step"] for h in history], 0.7)


def dodge(ys, gap):
    """Spread end-of-line labels so they don't overlap."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for i in range(1, len(order)):
        out[order[i]] = max(out[order[i]], out[order[i - 1]] + gap)
    return out


def legend_handles():
    names = {3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8+"}
    return [Patch(fc=SIDE_COLORS[k], ec="#1b1b1b", lw=0.6, label=names[k]) for k in range(3, 9)]


def network_polygons(net):
    """Unwrapped polygons, with copies across the periodic edges, and colours."""
    ang = corner_angles(net)
    t = net.topo
    at_junction = np.bincount(t.he_v, minlength=len(net.X))[t.he_v] == 3
    corners = np.bincount(t.he_cell, (ang < CORNER_CUT) & at_junction, net.ncell).astype(int)
    polys, cols = [], []
    for c, k in zip(net.cells, np.clip(corners, 3, 8)):
        P = unwrap(net.X[c])
        for dx in (-L, 0, L):
            for dy in (-L, 0, L):
                Q = P + (dx, dy)
                if Q[:, 0].max() > 0 and Q[:, 0].min() < L and Q[:, 1].max() > 0 and Q[:, 1].min() < L:
                    polys.append(Q)
                    cols.append(SIDE_COLORS[k])
    return polys, cols


def t_junctions(net):
    """Positions of junctions that still look like a T."""
    t = net.topo
    largest = np.full(len(net.X), -np.inf)
    np.maximum.at(largest, t.he_v, corner_angles(net))
    junction = np.bincount(t.he_v, minlength=len(net.X)) == 3
    return net.X[(largest > T_LIKE) & junction]


def draw_network(ax, net, show_T=True):
    polys, cols = network_polygons(net)
    coll = PolyCollection(polys, facecolors=cols, edgecolors="#1b1b1b", linewidths=0.9,
                          alpha=0.9)
    ax.add_collection(coll)
    T = t_junctions(net) if show_T else np.empty((0, 2))
    dots, = ax.plot(T[:, 0], T[:, 1], "o", ms=3.2, mfc="white", mec="#1b1b1b", mew=0.8)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK2)
    return coll, dots


def angle_histogram(ax, angles, ncell):
    h, _ = np.histogram(angles, ANGLE_BINS)
    bars = ax.bar(ANGLE_BINS[:-1], h, width=np.diff(ANGLE_BINS), align="edge",
                  color="#2a78d6", edgecolor="white", lw=0.5)
    for x, lab in ((90, "90° (T)"), (120, "120° (Y)"), (180, "180° (T)")):
        ax.axvline(x, color=INK2, ls="--", lw=0.8)
        ax.text(x, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=INK2)
    ax.set_xlim(60, 200)
    ax.set_ylim(0, 2.1 * ncell)
    ax.set_xlabel("angle between cracks at a junction (deg)", fontsize=8, color=INK2)
    style(ax)
    return bars


# ----------------------------------------------------------------------------
# The live dashboard
# ----------------------------------------------------------------------------
class Dashboard:
    """Network + three charts.  While the surface is still cracking it shows
    the cracks growing; afterwards, the columns and their statistics.

    rect = (left, bottom, width, height) of the figure area to use."""

    def __init__(self, fig, sim, rect=(0, 0, 1, 1), show_T=True):
        self.fig, self.show_T = fig, show_T
        l, b, w, h = rect
        X = lambda x: l + x * w            # noqa: E731  (map 0..1 into rect)
        Y = lambda y: b + y * h            # noqa: E731
        self.title = fig.text(X(0.0), Y(0.965), "", fontsize=14, color=INK, weight="bold")
        self.status = fig.text(X(0.0), Y(0.925), "", fontsize=10, color=INK2)
        self.ax_net = fig.add_axes([X(0.0), Y(0.03), 0.50 * w, 0.86 * h])
        self.note = fig.text(X(0.0), Y(-0.005), "", fontsize=8, color=INK2)
        self.ax_hist = fig.add_axes([X(0.64), Y(0.67), 0.34 * w, 0.22 * h])
        self.ax_sides = fig.add_axes([X(0.64), Y(0.37), 0.34 * w, 0.19 * h])
        self.ax_E = fig.add_axes([X(0.64), Y(0.07), 0.34 * w, 0.19 * h])
        self.reset(sim)

    def reset(self, sim):
        """(Re)build all artists for a new simulation, or a new phase of one."""
        for ax in (self.ax_net, self.ax_hist, self.ax_sides, self.ax_E):
            ax.cla()
            ax.set_axis_on()
        self.sim = sim
        self.mode = "forming" if sim.forming else "depth"
        if self.mode == "forming":
            self._build_forming()
        else:
            self._build_depth()
        self.update()

    def update(self):
        if (self.mode == "forming") != self.sim.forming:
            return self.reset(self.sim)            # the surface has finished cracking
        if self.mode == "forming":
            self._update_forming()
        else:
            self._update_depth()

    # ------------------------------------------------------------ surface cracking
    def _build_forming(self):
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
        ax = self.ax_net
        self.cracks = LineCollection([], colors="#1b1b1b", linewidths=1.0)
        ax.add_collection(self.cracks)
        self.tips, = ax.plot([], [], "o", ms=3.0, color=SIDE_COLORS[4], mec="none")
        self.Tdots, = ax.plot([], [], "o", ms=3.4, mfc="white", mec="#1b1b1b", mew=0.8)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f4f1ea")
        for sp in ax.spines.values():
            sp.set_color(INK2)
        ax.legend(handles=[Line2D([], [], color="#1b1b1b", lw=1.2, label="crack"),
                           Line2D([], [], ls="", marker="o", ms=5, color=SIDE_COLORS[4],
                                  label="growing tip"),
                           Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec="#1b1b1b",
                                  label="T junction")],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
        self.title.set_text("The cooling surface cracks (depth 0)")
        self.note.set_text("each crack stops where it reaches an older one, "
                           "making a T junction")
        h = self.ax_hist
        self.meet_bins = np.arange(0, 91, 5)
        self.meet_bars = h.bar(self.meet_bins[:-1], np.zeros(len(self.meet_bins) - 1), width=5,
                               align="edge", color="#2a78d6", edgecolor="white", lw=0.5)
        h.axvline(90, color=INK2, ls="--", lw=0.8)
        h.text(90, 1.0, "90°", transform=h.get_xaxis_transform(), ha="center", va="bottom",
               fontsize=7.5, color=INK2)
        h.set_xlim(0, 92)
        h.set_xlabel("angle at which a crack meets an older crack (deg)", fontsize=8, color=INK2)
        h.set_title("How new cracks meet old ones", fontsize=10, color=INK, loc="left", pad=14)
        style(h)
        s = self.ax_sides
        self.n_cr, = s.plot([], [], color=INK, lw=2, label="cracks")
        self.n_T, = s.plot([], [], color="#2a78d6", lw=2, label="T junctions")
        s.legend(fontsize=8, frameon=False, loc="upper left")
        s.set_title("Cracks and T junctions formed", fontsize=10, color=INK, loc="left")
        s.set_xlabel("time step at the surface", fontsize=8, color=INK2)
        style(s)
        self.ax_E.set_axis_off()
        self.ax_E.text(0.0, 0.5, "The depth charts start once the surface\n"
                       "has finished cracking.", fontsize=9, color=INK2,
                       transform=self.ax_E.transAxes, va="center")

    def _update_forming(self):
        g = self.sim.growth
        self.cracks.set_segments(crack_polylines(g))
        tips = np.array([g.cracks[i].arms[a][-1] for i, a in g.active_tips]).reshape(-1, 2) % L
        self.tips.set_data(tips[:, 0], tips[:, 1])
        J = np.array(g.junction_pts).reshape(-1, 2)
        self.Tdots.set_data(J[:, 0], J[:, 1])
        counts, _ = np.histogram(g.meet_angles, self.meet_bins)
        for bar, v in zip(self.meet_bars, counts):
            bar.set_height(v)
        self.ax_hist.set_ylim(0, max(10, 1.15 * counts.max()))
        hist = np.array(g.history).reshape(-1, 4)
        if len(hist):
            self.n_cr.set_data(hist[:, 0], hist[:, 1])
            self.n_T.set_data(hist[:, 0], hist[:, 2])
            self.ax_sides.set_xlim(0, max(60, hist[-1, 0] * 1.05))
            self.ax_sides.set_ylim(0, 2.15 * g.N)
        self.status.set_text(
            f"time step {g.step_no}    cracks {len(g.cracks)} / {g.N}    "
            f"T junctions {g.n_junctions}    growing tips {len(g.active_tips)}")

    # ------------------------------------------------------------ columns at depth
    def _build_depth(self):
        sim, net = self.sim, self.sim.net
        self.coll, self.dots = draw_network(self.ax_net, net, self.show_T)
        self.ax_net.legend(handles=legend_handles(), title="corners per\ncolumn",
                           loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                           title_fontsize=8, frameon=False)
        self.title.set_text("Cooling-crack network, cracks advancing downward")
        self.note.set_text("white dots = junctions still T-shaped (largest angle > 150°)")
        m = measure(net, sim.p.gamma)
        self.bars = angle_histogram(self.ax_hist, m["angles"], net.ncell)
        self.ax_hist.set_title("Angles at junctions", fontsize=10, color=INK, loc="left", pad=14)

        self.side_lines, self.side_labels = {}, {}
        for n in (4, 5, 6, 7):
            self.side_lines[n], = self.ax_sides.plot([], [], color=SIDE_COLORS[n], lw=2)
            self.side_labels[n] = self.ax_sides.text(0, 0, str(n), fontsize=8, color=INK,
                                                     va="center")
        self.ax_sides.set_ylim(0, 1)
        self.ax_sides.set_title("Share of columns with 4 / 5 / 6 / 7 corners", fontsize=10,
                                color=INK, loc="left")
        style(self.ax_sides)

        self.E_line, = self.ax_E.plot([], [], color=INK, lw=2)
        for name, (j, _) in REFERENCE.items():
            self.ax_E.axhline(j, color=INK2, ls="--", lw=0.8)
            self.ax_E.text(1, j, f"{name} {j:.4f} ", fontsize=7.5, color=INK2, ha="right",
                           va="bottom", transform=self.ax_E.get_yaxis_transform())
        self.ax_E.set_title("Elastic energy per column ⟨J/A²⟩", fontsize=10, color=INK,
                            loc="left")
        self.ax_E.set_xlabel("depth step (log scale)", fontsize=8, color=INK2)
        style(self.ax_E)

    def _update_depth(self):
        sim, net = self.sim, self.sim.net
        polys, cols = network_polygons(net)
        self.coll.set_verts(polys)
        self.coll.set_facecolor(cols)
        T = t_junctions(net) if self.show_T else np.empty((0, 2))
        self.dots.set_data(T[:, 0], T[:, 1])

        m = measure(net, sim.p.gamma)
        h, _ = np.histogram(m["angles"], ANGLE_BINS)
        for bar, v in zip(self.bars, h):
            bar.set_height(v)
        self.ax_hist.set_ylim(0, 2.1 * net.ncell)

        hist = sim.history
        x = depth_x(hist)
        for ax in (self.ax_sides, self.ax_E):
            log_depth_axis(ax, sim.p.steps)
        ends = dodge([hist[-1]["frac"][n] for n in (4, 5, 6, 7)], 0.09)
        for n, ye in zip((4, 5, 6, 7), ends):
            self.side_lines[n].set_data(x, [r["frac"][n] for r in hist])
            self.side_labels[n].set_position((x[-1] * 1.12, ye))
        jy = [r["shape_J"] for r in hist]
        self.E_line.set_data(x, jy)
        self.ax_E.set_ylim(0.158, max(0.17, min(0.3, max(jy) * 1.02)))

        last = hist[-1]
        self.status.set_text(
            f"depth step {sim.step_no} / {sim.p.steps}    columns {net.ncell}    "
            f"hexagons {last['frac'][6]:.0%}    T-junctions {last['frac_T']:.0%}    "
            f"neighbour swaps {net.n_T1}    pinch-outs {net.n_T2}")


def crack_polylines(growth):
    """Every crack as a polyline, with copies across the periodic edges."""
    out = []
    for c in growth.cracks:
        P = np.vstack([np.array(c.arms[0])[::-1], np.array(c.arms[1])[1:]])
        if len(P) < 2:
            continue
        base = np.floor(P[0] / L) * L
        P = P - base
        lo, hi = P.min(0), P.max(0)
        for dx in (-L, 0, L):
            for dy in (-L, 0, L):
                if hi[0] + dx > 0 and lo[0] + dx < L and hi[1] + dy > 0 and lo[1] + dy < L:
                    out.append(P + (dx, dy))
    return out


# ----------------------------------------------------------------------------
# Static figures
# ----------------------------------------------------------------------------
def plot_snapshots(path, snaps):
    """snaps: list of (step, CrackNetwork) pairs."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, len(snaps), figsize=(4.2 * len(snaps), 7.6),
                            gridspec_kw=dict(height_ratios=[3, 1.25]), dpi=130)
    for j, (k, net) in enumerate(snaps):
        draw_network(axs[0, j], net)
        m = measure(net)
        axs[0, j].set_title(f"depth step {k}\n{(m['corners'] == 6).mean():.0%} hexagons, "
                            f"{(m['largest_angle'] > T_LIKE).mean():.0%} T-junctions",
                            fontsize=10.5, color=INK)
        angle_histogram(axs[1, j], m["angles"], net.ncell)
    axs[0, -1].legend(handles=legend_handles(), title="corners", loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=8, frameon=False)
    fig.suptitle("From T junctions (90°/180°) to Y junctions (120°): hexagons emerge with depth",
                 fontsize=13, color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_summary(path, history):
    import matplotlib.pyplot as plt
    x = depth_x(history)
    steps = history[-1]["step"]
    g = history[-1]["gamma"]
    col = lambda key: [r[key] for r in history]          # noqa: E731
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), dpi=120)

    ax = axs[0, 0]
    ax.plot(x, col("largest_angle"), color=INK, lw=2)
    for y, lab in ((180, "T junction (90/90/180)"), (120, "Y junction (120/120/120)")):
        ax.axhline(y, color=INK2, ls="--", lw=0.8)
        ax.text(1, y + 1.5, lab, fontsize=8, color=INK2)
    ax.set_title("(a) Largest angle at each junction (mean)", loc="left", fontsize=10.5)
    ax.set_ylabel("degrees")

    ax = axs[0, 1]
    ax.plot(x, col("junctions_mean"), color="#2a78d6", lw=2)
    ax.plot(x, col("corners_mean"), color="#eb6834", lw=2)
    ax.text(x[-1], history[-1]["junctions_mean"] + 0.12, "junctions per column (Euler: always 6)",
            ha="right", fontsize=8.5, color=INK)
    ax.text(1, history[0]["corners_mean"] - 0.3, "real corners per column", fontsize=8.5, color=INK)
    ax.set_ylim(3.5, 6.6)
    ax.set_title("(b) Average sides: the hidden 6 becomes visible", loc="left", fontsize=10.5)

    ax = axs[0, 2]
    ends = dodge([history[-1]["frac"][n] for n in (4, 5, 6, 7)], 0.05)
    for n, ye in zip((4, 5, 6, 7), ends):
        ax.plot(x, [r["frac"][n] for r in history], color=SIDE_COLORS[n], lw=2)
        ax.text(x[-1] * 1.15, ye, f"{n}-sided", fontsize=8.5, va="center", color=INK)
    ax.set_ylim(0, 1)
    ax.set_title("(c) Share of columns by number of corners", loc="left", fontsize=10.5)

    for ax, key, idx, title in ((axs[1, 0], "shape_J", 0, "(d) Elastic energy shape factor ⟨J/A²⟩"),
                                (axs[1, 1], "shape_P", 1, "(e) Crack length per column ⟨P/√A⟩")):
        ax.plot(x, col(key), color=INK, lw=2)
        for name, ref in REFERENCE.items():
            ax.axhline(ref[idx], color=INK2, ls="--", lw=0.8)
            ax.text(x[-1], ref[idx], f"{name} = {ref[idx]:.4g} ", ha="right", va="bottom",
                    fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5)

    ax = axs[1, 2]
    ax.plot(x, col("energy"), color=INK, lw=2)
    for name, (j, p) in REFERENCE.items():           # perfect tiling: J/A² + gamma (P/√A)/2
        e = j + g * p / 2
        ax.axhline(e, color=INK2, ls="--", lw=0.8)
        ax.text(x[-1], e, f"perfect {name} tiling = {e:.4f} ", ha="right", va="bottom",
                fontsize=8, color=INK2)
    ax.set_title("(f) Total energy per column (what is minimised)", loc="left", fontsize=10.5)

    for ax in axs.ravel():
        log_depth_axis(ax, steps)
        ax.set_xlabel("depth step (log scale)", fontsize=8.5, color=INK2)
        style(ax)
    axs[0, 2].set_xlim(0.7, steps * 3)
    fig.suptitle("Nothing here is told about hexagons - every curve is measured from the "
                 "simulated network", fontsize=12.5, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_formation(path, snaps):
    """snaps: list of (time step, CrackGrowth copy).  Last panel: meeting angles."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    n = len(snaps)
    fig, axs = plt.subplots(1, n + 1, figsize=(4.0 * (n + 1), 4.6), dpi=130,
                            gridspec_kw=dict(width_ratios=[1] * n + [1.15]))
    for ax, (k, g) in zip(axs, snaps):
        ax.add_collection(LineCollection(crack_polylines(g), colors="#1b1b1b", linewidths=0.8))
        tips = np.array([g.cracks[i].arms[a][-1] for i, a in g.active_tips]).reshape(-1, 2) % L
        ax.plot(tips[:, 0], tips[:, 1], "o", ms=2.6, color=SIDE_COLORS[4], mec="none")
        J = np.array(g.junction_pts).reshape(-1, 2)
        ax.plot(J[:, 0], J[:, 1], "o", ms=2.8, mfc="white", mec="#1b1b1b", mew=0.6)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#f4f1ea")
        ax.set_title(f"time step {k}\n{len(g.cracks)} cracks, {g.n_junctions} T junctions",
                     fontsize=10, color=INK)
    h = axs[-1]
    g = snaps[-1][1]
    h.hist(g.meet_angles, bins=np.arange(0, 91, 5), color="#2a78d6", edgecolor="white", lw=0.5)
    h.axvline(90, color=INK2, ls="--", lw=0.8)
    h.set_xlim(0, 92)
    h.set_xlabel("angle at which a crack meets an older one (deg)", fontsize=8.5, color=INK2)
    h.set_title("Meeting angles (all junctions)", fontsize=10, color=INK)
    style(h)
    fig.suptitle("The surface cracks from intact crust: every junction forms as a T",
                 fontsize=13, color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_gif(path, frames, first_ms=1600, ms=90, last_ms=3500):
    from PIL import Image
    picks = [frames[0], frames[len(frames) // 3], frames[-1]]
    sheet = Image.new("RGB", (picks[0].width, picks[0].height * 3))
    for i, f in enumerate(picks):
        sheet.paste(f, (0, i * f.height))
    pal = sheet.convert("P", palette=Image.ADAPTIVE, colors=128)
    imgs = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    dur = [first_ms] + [ms] * (len(imgs) - 2) + [last_ms]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0,
                 optimize=True)


def fig_to_image(fig):
    from PIL import Image
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
