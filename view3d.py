"""
view3d.py - the columns in 3D, in a window you can rotate.
==========================================================

    python view3d.py                       # default run (300 columns, 300 depth steps)
    python view3d.py --cells 200 --seed 7 --steps 600
    python view3d.py --window 1.0          # show the whole patch instead of a part of it
    python view3d.py --box                 # cut the block into a neat box
    python view3d.py --gif basalt_3d.gif   # no window: a GIF of the columns growing

Needs:  pip install pyvista

How the 3D block is made
    The simulation follows the crack pattern slice by slice as the cracks grow
    downward (simulation.py).  Every few depth steps the column outlines are
    recorded.  Between two recorded slices, each corner of a column is joined
    to the same corner one slice lower, so the column walls are continuous
    surfaces.  Where a column changes its number of corners (a T1 swap, a
    neighbour pinching out) there is no corner-to-corner match, so the column
    takes a small step there instead.  Columns are drawn a little narrower
    than they are, so the cracks between them show as gaps.

    Colour = number of real corners of the column at that depth (the same
    colours as the 2D plots): red 3, orange 4, yellow 5, blue 6, violet 7.

Speed
    The simulation result is saved (in ~/.cache/basalt_sim), so opening the
    same settings again skips the simulation; --no-cache turns this off.
    The whole block is one mesh.  The sliders only choose which part of it
    is drawn, by cutting arrays that were prepared once.

Controls in the window
    left-drag rotate          right-drag / wheel  zoom        middle-drag  pan
    "grown down to" slider    where the crack front has reached (bottom face)
    "cut from the top" slider removes the upper part, so the top face shows
                              the cross-section at that depth, seen from above
    Play button or space      grow the columns down from the surface
    r  reset view     b  view from below     s  save a screenshot     q  quit
"""

import argparse
import hashlib
import os
import pickle

import numpy as np

from measure import measure
from physics import unwrap, minimg
from plots import SIDE_COLORS
from simulation import Simulation, Params

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "basalt_sim")
CACHE_VERSION = 2          # raise when the simulation changes, so old results are not reused


# ----------------------------------------------------------------------------
# 1) Record the column outlines while the cracks grow downward
# ----------------------------------------------------------------------------
def record_slices(p: Params, every=3, progress=True):
    """Run the simulation and keep the column outlines every `every` steps."""
    sim = Simulation(p, watch_forming=False)
    slices = []

    def keep():
        net = sim.net
        m = measure(net, p.gamma)
        h = sim.history[-1]
        slices.append(dict(
            step=sim.step_no,
            ids=list(net.cell_ids),
            polys=[unwrap(net.X[c]) for c in net.cells],
            corners=np.clip(m["corners"], 3, 8),
            hexagons=h["frac"][6], t_junctions=h["frac_T"], largest=h["largest_angle"]))

    keep()
    while sim.step_no < p.steps:
        sim.advance(min(every, p.steps - sim.step_no), record_every=every)
        keep()
        if progress and len(slices) % 20 == 0:
            print(f"  depth step {sim.step_no} / {p.steps}")
    return slices


def cached_slices(p: Params, every, use_cache=True, progress=True):
    """record_slices, but reuse the result of an earlier run with the same settings."""
    key = repr((CACHE_VERSION, p.pattern, p.n_cells, p.seed, p.gamma, p.dt, p.max_move,
                p.steps, every))
    path = os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest()[:16] + ".pkl")
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                saved_key, slices = pickle.load(f)
            if saved_key == key:
                if progress:
                    print("using the saved simulation result (--no-cache to rerun it)")
                return slices
        except Exception:
            pass
    if progress:
        print(f"growing the cracks down {p.steps} steps (outline kept every {every}) ...")
    slices = record_slices(p, every, progress)
    if use_cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump((key, slices), f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass
    return slices


# ----------------------------------------------------------------------------
# 2) Geometry: place, shrink, (clip), and join the slices into columns
# ----------------------------------------------------------------------------
def _clip(P, x0, x1, y0, y1):
    """Sutherland-Hodgman clipping of polygon P to an axis-aligned box."""
    for axis, bound, keep_low in ((0, x0, False), (0, x1, True), (1, y0, False), (1, y1, True)):
        if len(P) == 0:
            break
        Q = []
        for i in range(len(P)):
            a, b = P[i - 1], P[i]
            ina = a[axis] <= bound if keep_low else a[axis] >= bound
            inb = b[axis] <= bound if keep_low else b[axis] >= bound
            if inb:
                if not ina:
                    Q.append(a + (b - a) * (bound - a[axis]) / (b[axis] - a[axis]))
                Q.append(b)
            elif ina:
                Q.append(a + (b - a) * (bound - a[axis]) / (b[axis] - a[axis]))
        P = np.array(Q)
    return P


def column_outline(poly, window, shrink, clip=False):
    """The column outline, shrunk, placed at its periodic copy nearest the
    window centre, counter-clockwise, in window coordinates (or None).  With
    clip=True it is cut to the window (a neat box); otherwise whole columns
    are kept, giving a natural, ragged edge."""
    x0, y0, w = window
    centre = np.array([x0 + w / 2, y0 + w / 2])
    A, c = _area_centroid(poly)
    shift = minimg(c - centre) - (c - centre)
    P = c + shift + shrink * (poly - c)
    if A < 0:
        P = P[::-1]
    if clip:
        P = _clip(P, x0, x0 + w, y0, y0 + w)
    return P - (x0, y0) if len(P) >= 3 else None


def _area_centroid(P):
    """Signed area and centroid of a polygon (shoelace formula)."""
    x, y = P[:, 0], P[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = cr.sum() / 2
    return A, np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (6 * A)


def members(slice0, window):
    """Columns whose centre at the surface lies inside the window."""
    x0, y0, w = window
    centre = np.array([x0 + w / 2, y0 + w / 2])
    keep = set()
    for cid, poly in zip(slice0["ids"], slice0["polys"]):
        c = _area_centroid(poly)[1]
        if np.abs(minimg(c - centre)).max() < w / 2:
            keep.add(cid)
    return keep


def match_corners(P, Q, tol):
    """If Q is P one slice lower (same number of corners, each moved less than
    tol), return the shift r such that corner i of P continues as corner
    (i + r) % m of Q.  Otherwise None."""
    m = len(P)
    if len(Q) != m:
        return None
    shifted = (np.arange(m)[:, None] + np.arange(m)[None, :]) % m     # row r: i + r
    d = np.hypot(*(Q[shifted] - P[None]).transpose(2, 0, 1))         # (shift, corner)
    r = int(np.argmin(d.sum(axis=1)))
    return r if d[r].max() < tol else None


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


COLORS = {k: _rgb(v) for k, v in SIDE_COLORS.items()}


def _line_arrays(polys, z):
    """Closed outlines at height z as (points, flat VTK lines array, mask of
    the entries of that array that are counts rather than point numbers)."""
    if not polys:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, bool)
    pts, lines, counts, b = [], [], [], 0
    for Q in polys:
        m = len(Q)
        pts.append(np.column_stack((Q, np.full(m, z))))
        lines.append(np.concatenate(([m + 1], b + np.arange(m), [b])))
        counts.append(np.arange(m + 2) == 0)
        b += m
    return np.vstack(pts), np.concatenate(lines), np.concatenate(counts)


class Block:
    """All the geometry of the 3D block, prepared once.

    Recorded slice k lies at height z_k = -k * dz.  "Band" k is the layer
    between slice k and slice k+1.  For every band we store the faces of the
    column walls (all bands in one flat VTK face array), and separately the
    faces that close the block when that band is the top or the bottom of
    what is shown.  solid() and outlines() then only cut and join arrays.
    """

    def __init__(self, slices, window=(0.0, 0.0, 0.5), height=0.5, shrink=0.9, clip=False):
        show = None if clip else members(slices[0], window)
        outl = []                               # per slice: {id: (outline, corners)}
        for s in slices:
            d = {}
            for cid, poly, corners in zip(s["ids"], s["polys"], s["corners"]):
                if show is not None and cid not in show:
                    continue
                Q = column_outline(poly, window, shrink, clip)
                if Q is not None:
                    d[cid] = (Q, int(corners))
            outl.append(d)
        self.n = max(len(slices) - 1, 1)        # number of bands
        dz = height / self.n
        z = -np.arange(len(slices)) * dz
        # a corner may move at most this far between slices to be joined
        tol = 0.3 * shrink * np.sqrt(1.0 / max(len(slices[0]["ids"]), 1))

        pts, n_pts = [], 0

        def add_points(Q, zz):
            nonlocal n_pts
            pts.append(np.column_stack((Q, np.full(len(Q), zz))))
            n_pts += len(Q)
            return n_pts - len(Q)

        # every outline of every slice gets its points once
        base = [{cid: add_points(Q, z[k]) for cid, (Q, _) in d.items()}
                for k, d in enumerate(outl)]

        walls, walls_rgb = [], []                # per band
        self.top_caps, self.bot_caps = [], []    # per band: (faces, colours)
        self.top_lines, self.bot_lines = [], []  # per band: outlines on those faces
        joined_above = set()                     # ids joined corner-to-corner from above
        for k in range(self.n):
            F, C, TF, TC, BF, BC, TL, BL = [], [], [], [], [], [], [], []
            joined = set()
            for cid, (P, corners) in outl[k].items():
                m, b0, rgb = len(P), base[k][cid], COLORS[corners]
                ring = np.arange(m)
                top_face = np.concatenate(([m], b0 + ring))
                TF.append(top_face)              # closes the block if band k is the top
                TC.append(rgb)
                TL.append(P)
                if cid not in joined_above:      # the column steps here: close it
                    F.append(top_face)
                    C.append(rgb)
                below = outl[k + 1].get(cid)
                r = None if below is None else match_corners(P, below[0], tol)
                if r is not None:                # continuous wall down to slice k+1
                    b1 = base[k + 1][cid]
                    lower, lower_next = b1 + (ring + r) % m, b1 + (ring + 1 + r) % m
                    joined.add(cid)
                    BF.append(np.concatenate(([m], lower[::-1])))
                    BC.append(COLORS[below[1]])
                    BL.append(below[0])
                else:                            # straight wall, closed underneath
                    b1 = add_points(P, z[k + 1])
                    lower, lower_next = b1 + ring, b1 + (ring + 1) % m
                    F.append(np.concatenate(([m], lower[::-1])))
                    C.append(rgb)
                    BL.append(P)
                quads = np.column_stack((np.full(m, 4), b0 + ring, b0 + (ring + 1) % m,
                                         lower_next, lower))
                F.append(quads.ravel())
                C.extend([rgb] * m)
            joined_above = joined
            walls.append(np.concatenate(F) if F else np.zeros(0, int))
            walls_rgb.append(np.array(C, np.uint8).reshape(-1, 3))
            self.top_caps.append((np.concatenate(TF) if TF else np.zeros(0, int),
                                  np.array(TC, np.uint8).reshape(-1, 3)))
            self.bot_caps.append((np.concatenate(BF) if BF else np.zeros(0, int),
                                  np.array(BC, np.uint8).reshape(-1, 3)))
            self.top_lines.append(_line_arrays(TL, z[k] + 2e-4))
            self.bot_lines.append(_line_arrays(BL, z[k + 1] - 2e-4))

        self.points = np.vstack(pts)
        self.faces = np.concatenate(walls)
        self.rgb = np.vstack(walls_rgb)
        self.face_off = np.r_[0, np.cumsum([len(f) for f in walls])]
        self.cell_off = np.r_[0, np.cumsum([len(c) for c in walls_rgb])]
        self.dz = dz

    def solid(self, top, bottom):
        """Faces and colours of bands top .. bottom-1, closed at both ends."""
        (tf, tc), (bf, bc) = self.top_caps[top], self.bot_caps[bottom - 1]
        f0, f1 = self.face_off[top], self.face_off[bottom]
        c0, c1 = self.cell_off[top], self.cell_off[bottom]
        return (np.concatenate([self.faces[f0:f1], tf, bf]),
                np.vstack([self.rgb[c0:c1], tc, bc]))

    def outlines(self, top, bottom):
        """Points and lines of the outlines on the top and the bottom face."""
        (p1, l1, _), (p2, l2, is_count) = self.top_lines[top], self.bot_lines[bottom - 1]
        return np.vstack([p1, p2]), np.concatenate([l1, np.where(is_count, l2, l2 + len(p1))])


# ----------------------------------------------------------------------------
# 3) The window
# ----------------------------------------------------------------------------
class Viewer:
    """Bands top..bottom-1 are shown.  'Grown down to' moves the bottom (the
    crack front); 'cut from the top' removes the upper part, so the top face
    becomes the cross-section at that depth, seen from above."""

    def __init__(self, slices, p, window=(0.0, 0.0, 0.5), height=0.5, shrink=0.9,
                 clip=False, off_screen=False, widgets=True, window_size=(1250, 860)):
        import pyvista as pv
        self.pv, self.slices, self.p = pv, slices, p
        print("building the 3D columns ...")
        self.block = Block(slices, window, height, shrink, clip)
        self.n = self.block.n
        self.top, self.bottom = 0, self.n
        self.playing = False

        pl = pv.Plotter(window_size=list(window_size), off_screen=off_screen,
                        title="Basalt columns in 3D")
        # cheap edge smoothing instead of 8x multisampling: much faster on
        # laptops and high-resolution screens
        pl.disable_anti_aliasing()
        pl.enable_anti_aliasing("fxaa")
        if widgets:
            pl.set_background("#f4f3f0", top="#dfe6ee")
        else:                                    # plain: GIF colours stay true
            pl.set_background("#f4f3f0")
        self.pl = pl

        # one mesh for all the columns, one for the outlines; the sliders only
        # change which faces they contain
        self.mesh = pv.PolyData()
        self.mesh.points = self.block.points       # no point cells, only faces
        self.lines = pv.PolyData()
        self._update_meshes()
        pl.add_mesh(self.mesh, scalars="rgb", rgb=True, show_scalar_bar=False)
        pl.add_mesh(self.lines, color="#1b1b1b", line_width=1.0)

        pl.add_text("Basalt columns in 3D", position=(0.015, 0.955), viewport=True,
                    font_size=15, color="#222222")
        self.readout = pl.add_text(" ", position=(0.015, 0.865 if widgets else 0.85),
                                   viewport=True, font_size=10 if widgets else 12,
                                   color="#333333")
        pl.add_legend([[lab, SIDE_COLORS[k]] for k, lab in
                       ((3, "3 corners"), (4, "4 corners"), (5, "5 corners"),
                        (6, "6 corners (hexagon)"), (7, "7 corners"), (8, "8 or more"))],
                      bcolor="white", border=True, face="rectangle",
                      size=(0.17, 0.2) if widgets else (0.25, 0.22),
                      loc="upper right" if widgets else "lower right")
        self.s_bottom = self.s_top = None
        self.widgets = widgets
        if widgets:
            pl.add_text("drag: rotate    wheel: zoom    middle-drag: pan\n"
                        "Play / space: grow down from the surface\n"
                        "upper slider: cut away the top part\n"
                        "r: reset view   b: from below   s: screenshot",
                        position=(0.015, 0.075), viewport=True, font_size=8, color="#555555")
            self.s_bottom = pl.add_slider_widget(
                self._on_bottom, rng=[1, self.n], value=self.n, title="grown down to",
                pointa=(0.42, 0.075), pointb=(0.82, 0.075), fmt="%.0f",
                interaction_event="always", style="modern", title_height=0.02)
            self.s_top = pl.add_slider_widget(
                self._on_top, rng=[0, self.n - 1], value=0, title="cut from the top",
                pointa=(0.42, 0.175), pointb=(0.82, 0.175), fmt="%.0f",
                interaction_event="always", style="modern", title_height=0.02)
            pl.add_checkbox_button_widget(self._on_play, value=False, position=(20, 20),
                                          size=36, color_on="#2a78d6", color_off="#9aa5b1")
            pl.add_text("Play", position=(64, 30), font_size=10, color="#333333")
            pl.add_key_event("space", lambda: self._on_play(not self.playing))
            pl.add_key_event("s", self._screenshot)
            pl.add_key_event("r", self._reset_view)
            pl.add_key_event("b", self._view_below)
            pl.add_timer_event(max_steps=10 ** 7, duration=50, callback=self._on_timer)
        self._reset_view()
        self._show()

    # -------------------------------------------------------------- showing
    def _update_meshes(self):
        faces, rgb = self.block.solid(self.top, self.bottom)
        self.mesh.faces = faces
        self.mesh.cell_data["rgb"] = rgb
        pts, lines = self.block.outlines(self.top, self.bottom)
        self.lines.points = pts
        self.lines.lines = lines

    def _show(self):
        self._update_meshes()
        st, sb = self.slices[self.top], self.slices[self.bottom]
        if not self.widgets:                     # short version for the GIF
            self.readout.SetInput(
                f"crack front at depth step {sb['step']}\n"
                f"hexagons {sb['hexagons']:.0%}    T junctions {sb['t_junctions']:.0%}")
            return
        self.readout.SetInput(
            f"top face:     depth step {st['step']:4d}   hexagons {st['hexagons']:4.0%}   "
            f"T junctions {st['t_junctions']:4.0%}   largest angle {st['largest']:.0f}°\n"
            f"bottom face: depth step {sb['step']:4d}   hexagons {sb['hexagons']:4.0%}   "
            f"T junctions {sb['t_junctions']:4.0%}   largest angle {sb['largest']:.0f}°")

    def _on_bottom(self, value):
        self.bottom = int(np.clip(round(value), self.top + 1, self.n))
        self._show()

    def _on_top(self, value):
        self.top = int(np.clip(round(value), 0, self.bottom - 1))
        self._show()

    def set_view(self, top=None, bottom=None):
        """Move the sliders from code (used by Play, screenshots and the GIF)."""
        if bottom is not None:
            self.bottom = int(np.clip(bottom, self.top + 1, self.n))
            if self.s_bottom is not None:
                self.s_bottom.GetRepresentation().SetValue(self.bottom)
        if top is not None:
            self.top = int(np.clip(top, 0, self.bottom - 1))
            if self.s_top is not None:
                self.s_top.GetRepresentation().SetValue(self.top)
        self._show()

    def _on_play(self, state):
        self.playing = bool(state)
        if self.playing and self.bottom >= self.n:
            self.set_view(bottom=self.top + 1)

    def _on_timer(self, step=None):
        if not self.playing:
            return
        if self.bottom >= self.n:
            self.playing = False
            return
        self.set_view(bottom=self.bottom + 1)
        self.pl.render()

    def _camera(self, direction, zoom, shift):
        self.pl.view_vector(direction, viewup=(0, 0, 1))
        self.pl.camera.zoom(zoom)
        # move the picture up a little, clear of the sliders at the bottom
        self.pl.camera.SetWindowCenter(0.0, shift)
        self.pl.render()

    def _reset_view(self):
        self._camera((1.0, 0.85, 0.75), 0.85, -0.12)

    def _view_below(self):
        self._camera((1.0, 0.85, -0.9), 0.78, -0.06)

    def _screenshot(self, path=None):
        path = path or (f"basalt_3d_top{self.slices[self.top]['step']}"
                        f"_bottom{self.slices[self.bottom]['step']}.png")
        self.pl.render()
        self.pl.screenshot(path)
        print("saved", path)

    def frame(self):
        """The current picture as a PIL image."""
        from PIL import Image
        self.pl.render()
        return Image.fromarray(self.pl.screenshot(return_img=True))

    def show(self):
        self.pl.show()


# ----------------------------------------------------------------------------
# 4) A GIF of the columns growing down while the camera circles the block
# ----------------------------------------------------------------------------
def make_gif(viewer, path, turn=200.0, every=2, hold=30, ms=80):
    """Grow the block by `every` bands per frame while the camera turns by
    `turn` degrees in total, then keep turning for `hold` more frames."""
    from PIL import Image
    v = viewer
    v.set_view(top=0, bottom=v.n)
    v._camera((1.0, 0.85, 0.6), 0.8, 0.02)        # frame the whole block once
    v.pl.camera.SetWindowCenter(0.08, 0.02)       # a little left, clear of the legend
    bottoms = list(range(every, v.n + 1, every))
    if bottoms[-1] != v.n:
        bottoms.append(v.n)
    total = len(bottoms) + hold
    frames = []
    for i in range(total):
        v.set_view(bottom=bottoms[min(i, len(bottoms) - 1)])
        if i:
            v.pl.camera.Azimuth(turn / total)
        frames.append(v.frame())
        if i % 20 == 0:
            print(f"  frame {i + 1} / {total}")
    # one shared colour palette, taken from frames spread over the whole film
    picks = frames[len(frames) // 8::len(frames) // 8]
    w, h = frames[0].size
    sheet = Image.new("RGB", (w, h * len(picks)))
    for k, f in enumerate(picks):
        sheet.paste(f, (0, k * h))
    adaptive = sheet.quantize(colors=244, method=Image.Quantize.MEDIANCUT).getpalette()
    exact = [c for k in sorted(COLORS) for c in COLORS[k]] + [34, 34, 34, 27, 27, 27]
    pal = Image.new("P", (1, 1))
    pal.putpalette(adaptive[:244 * 3] + exact)  # + the legend colours, exactly
    imgs = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    dur = [900] + [ms] * (len(imgs) - 2) + [1500]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0,
                 optimize=True)
    print("saved", path)


def main():
    d = Params()
    ap = argparse.ArgumentParser(description="Basalt columns in a rotatable 3D window")
    ap.add_argument("--pattern", default=d.pattern)
    ap.add_argument("--cells", type=int, default=d.n_cells)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--gamma", type=float, default=d.gamma)
    ap.add_argument("--dt", type=float, default=d.dt)
    ap.add_argument("--steps", type=int, default=300, help="how deep to grow (depth steps)")
    ap.add_argument("--every", type=int, default=0,
                    help="record the outlines every this many steps (default: ~100 slices)")
    ap.add_argument("--window", type=float, default=0.5,
                    help="width of the part of the patch shown (1.0 = all of it)")
    ap.add_argument("--height", type=float, default=1.0,
                    help="height of the block, in units of the width shown")
    ap.add_argument("--box", action="store_true",
                    help="cut the columns at the window edges (a neat box)")
    ap.add_argument("--shrink", type=float, default=0.9,
                    help="columns are drawn at this fraction of their width")
    ap.add_argument("--screenshot", default="",
                    help="save a picture instead of opening a window")
    ap.add_argument("--gif", default="",
                    help="save a GIF of the columns growing instead of opening a window")
    ap.add_argument("--no-cache", action="store_true",
                    help="always rerun the simulation (do not reuse a saved result)")
    a = ap.parse_args()
    p = Params(pattern=a.pattern, n_cells=a.cells, seed=a.seed, gamma=a.gamma, dt=a.dt,
               steps=a.steps)
    every = a.every or max(1, round(a.steps / 100))
    slices = cached_slices(p, every, use_cache=not a.no_cache)
    w = min(max(a.window, 0.05), 1.0)
    window = ((1 - w) / 2, (1 - w) / 2, w)
    if a.gif:
        v = Viewer(slices, p, window, a.height * w, a.shrink, a.box, off_screen=True,
                   widgets=False, window_size=(760, 680))
        make_gif(v, a.gif)
        return
    v = Viewer(slices, p, window, a.height * w, a.shrink, a.box, off_screen=bool(a.screenshot))
    if a.screenshot:
        v._screenshot(a.screenshot)
    else:
        v.show()


if __name__ == "__main__":
    main()
