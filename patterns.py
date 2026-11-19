"""
patterns.py - starting crack patterns (the lava surface at depth 0).
====================================================================

"growth"    (default)  GROWING CRACKS from intact crust, no seed at all.
    Cracks start where the crust is most stressed, grow at both tips, turn
    to meet older cracks at ~90 degrees and stop there: every junction is
    created as a T junction.  This one is animated: see crack_growth.py.

"cracking"  SPLITTING PIECES -> T junctions
    A faster shortcut to a similar surface.  The piece that stores the most
    elastic energy breaks next, straight across its long axis (perpendicular
    to the largest tension), and the crack ends on the piece's sides: T
    junctions.  Needs a few first cracks to start from (16 random regions).

"voronoi"   RANDOM POINTS
    Take random points and give each the region closest to it.  Fully random,
    junctions at random angles, no T junctions at all.  Useful to show that
    the end state does not depend on starting from T junctions.

"brick"     the seed used in the first version: a few long, straight,
    horizontal cracks offset like a brick wall, then splitting.  The
    horizontal alignment is an artefact of that seed, kept for comparison.
"""

import numpy as np
from scipy.spatial import Voronoi, cKDTree

from physics import L, unwrap, polygon_moments
from network import CrackNetwork

PATTERNS = {
    "growth": "Growing cracks (no seed)",
    "cracking": "Splitting pieces",
    "voronoi": "Random points (Voronoi)",
    "brick": "Brick wall + splitting",
}
CRACKS_PER_COLUMN = 1.1     # growth: a few cracks end in slivers that are merged away


# ----------------------------------------------------------------------------
def periodic_voronoi(n, rng, tries=100):
    """Voronoi diagram of n random points on the periodic box."""
    for _ in range(tries):
        pts = rng.random((n, 2)) * L
        shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]) * L
        vor = Voronoi(np.vstack([pts + s for s in shifts]))
        centre = 4 * n                                  # index of the (0, 0) copy
        cells, raw = [], []
        for i in range(n):
            reg = vor.regions[vor.point_region[centre + i]]
            V = vor.vertices[reg]
            order = np.argsort(np.arctan2(V[:, 1] - pts[i, 1], V[:, 0] - pts[i, 0]))
            cells.append([len(raw) + k for k in range(len(reg))])
            raw.extend(V[order])
        raw = np.array(raw) % L
        raw[raw >= L] -= L
        # the same junction appears in up to 3 cells (possibly from a shifted copy)
        tree = cKDTree(raw, boxsize=L)
        canon = np.arange(len(raw))
        for a, b in tree.query_pairs(1e-9):
            canon[max(a, b)] = min(canon[a], canon[b])
        uniq, new_id = np.unique(canon, return_inverse=True)
        X = raw[uniq]
        cells = [[int(new_id[k]) for k in c] for c in cells]
        areas = np.array([polygon_moments(X[c])[0] for c in cells])
        ok = (np.all(areas > 0) and abs(areas.sum() - L * L) < 1e-9
              and all(np.ptp(unwrap(X[c]), axis=0).max() < 0.49 * L for c in cells))
        if ok and np.all(np.bincount([v for c in cells for v in c]) == 3):
            return X, cells
    raise RuntimeError("could not build a periodic Voronoi seed; use more cells")


# ----------------------------------------------------------------------------
def brick_seed(rng, rows=4, bricks=3):
    """A few long early cracks offset like a brick wall (all T junctions)."""
    h = rng.uniform(0.7, 1.3, rows)
    ys = (np.concatenate([[0], np.cumsum(h)[:-1]]) / h.sum() + rng.random()) % L
    X, bottoms, tops = [], [], []
    for r in range(rows):
        w = rng.uniform(0.7, 1.3, bricks)
        xs = np.sort((np.concatenate([[0], np.cumsum(w)[:-1]]) / w.sum() + rng.random()) % L)
        B, T = [], []
        for x in xs:
            X.append([x, ys[r]]); B.append(len(X) - 1)
            X.append([x, ys[(r + 1) % rows]]); T.append(len(X) - 1)
        bottoms.append((xs, B)); tops.append((xs, T))
    cells = []
    for r in range(rows):
        xs, B = bottoms[r]
        _, T = tops[r]
        xs_below, T_below = tops[(r - 1) % rows]
        xs_above, B_above = bottoms[(r + 1) % rows]
        for j in range(bricks):
            x0 = xs[j]
            width = (xs[(j + 1) % bricks] - x0) % L

            def inside(xx):
                d = (xx - x0) % L
                return [k for k in np.argsort(d) if 0 < d[k] < width]
            floor = [T_below[k] for k in inside(xs_below)]
            roof = [B_above[k] for k in inside(xs_above)][::-1]
            cells.append([B[j]] + floor + [B[(j + 1) % bricks], T[(j + 1) % bricks]]
                         + roof + [T[j]])
    return np.array(X), cells


# ----------------------------------------------------------------------------
def sequential_cracking(X, cells, n_target, rng, jitter_deg=6.0, select_power=2.0):
    """Keep breaking the most-stressed piece across its long axis."""
    nv = len(X)
    Xa = np.zeros((nv + 2 * (n_target - len(cells)), 2))   # room for all new junctions
    Xa[:nv] = X
    cells = [list(c) for c in cells]
    owner = {}                                       # (u, v) -> cell holding edge u->v
    for i, c in enumerate(cells):
        for k in range(len(c)):
            owner[(c[k], c[(k + 1) % len(c)])] = i
    J = np.zeros(n_target)
    J[:len(cells)] = [polygon_moments(Xa[c])[2] for c in cells]

    while len(cells) < n_target:
        n_now = len(cells)
        w = J[:n_now] ** select_power
        i = rng.choice(n_now, p=w / w.sum())        # most stored energy breaks first
        c = cells[i]
        P = unwrap(Xa[c])
        A, centre, _, C = polygon_moments(P)
        evals, evecs = np.linalg.eigh(C)
        u = evecs[:, 1]                             # long axis = largest tension
        th = np.radians(rng.normal(0, jitter_deg))
        u = np.array([np.cos(th) * u[0] - np.sin(th) * u[1],
                      np.sin(th) * u[0] + np.cos(th) * u[1]])
        half = 2 * np.sqrt(evals[1] / A)
        n = len(c)
        for _ in range(30):                         # crack line: across the long axis
            p0 = centre + u * rng.uniform(-0.25, 0.25) * half
            side = (P - p0) @ u
            hits = [k for k in range(n) if side[k] * side[(k + 1) % n] < 0]
            if len(hits) != 2:
                continue
            ts = [side[k] / (side[k] - side[(k + 1) % n]) for k in hits]
            if all(0.12 < t < 0.88 for t in ts):
                break
        else:
            J[i] *= 0.5                             # awkward piece; try others first
            continue

        new = []
        for k, t in zip(hits, ts):
            a, b = c[k], c[(k + 1) % n]
            Xa[nv] = (P[k] + t * (P[(k + 1) % n] - P[k])) % L
            new.append(nv)
            j = owner.pop((b, a))                   # the older crack's other side ...
            cj = cells[j]
            cj.insert(cj.index(a), nv)              # ... gets a T junction on it
            owner[(b, nv)] = owner[(nv, a)] = j
            nv += 1
        (k1, k2), (n1, n2) = hits, new
        for k in range(n):
            owner.pop((c[k], c[(k + 1) % n]), None)
        A_cell = [n1] + [c[(k1 + 1 + m) % n] for m in range((k2 - k1) % n)] + [n2]
        B_cell = [n2] + [c[(k2 + 1 + m) % n] for m in range((k1 - k2) % n)] + [n1]
        cells[i] = A_cell
        cells.append(B_cell)
        for idx, cc in ((i, A_cell), (n_now, B_cell)):
            for k in range(len(cc)):
                owner[(cc[k], cc[(k + 1) % len(cc)])] = idx
            J[idx] = polygon_moments(Xa[cc])[2]
    return Xa[:nv], cells


# ----------------------------------------------------------------------------
def make_growth(n_cells, seed):
    """The (not yet run) crack growth for the 'growth' start."""
    from crack_growth import CrackGrowth
    return CrackGrowth(int(round(CRACKS_PER_COLUMN * n_cells)), np.random.default_rng(seed))


def grown_network(growth, n_cells, seed, attempts=5):
    """Finish the growth and convert it to columns.  In the rare case that the
    cracks cannot be turned into proper columns, grow again with another draw."""
    from crack_growth import GrowthError
    for k in range(attempts):
        try:
            return growth.run().to_network()
        except GrowthError:
            growth = make_growth(n_cells, seed + 7919 * (k + 1))
    raise RuntimeError("crack growth failed repeatedly; try another seed")


def make_pattern(kind, n_cells, seed):
    rng = np.random.default_rng(seed)
    if kind == "growth":
        return grown_network(make_growth(n_cells, seed), n_cells, seed)
    if kind == "voronoi":
        X, cells = periodic_voronoi(n_cells, rng)
    elif kind == "brick":
        X, cells = sequential_cracking(*brick_seed(rng), n_cells, rng)
    elif kind == "cracking":
        X, cells = sequential_cracking(*periodic_voronoi(min(16, n_cells), rng), n_cells, rng)
    else:
        raise ValueError(f"unknown pattern {kind!r}; choose from {list(PATTERNS)}")
    return CrackNetwork(X, cells)
