"""
network.py - the crack network as a data structure, and how it changes.
=======================================================================

Junctions (vertices) have positions X.  Each column (cell) is a list of
junction ids in counter-clockwise order.  Neighbouring columns share junctions,
so moving one junction moves every crack that meets there.

Three things can happen as the crack front advances one depth step:

  relax_step : every junction slides a little down the energy gradient
               dX/dz = -mu dE/dX   (cracks cannot jump, so the move is capped)

  T1 swap    : a crack segment shrinks to ~0 and its two junctions reconnect
               the other way (columns P,Q stop touching, R,S start touching):

                    R                         R
                    |                        / \\
               P  --a--  Q     ->      P    a---b    Q      (rotated 90 deg)
                    |                        \\ /
                    b                         S
                    |
                    S

  T2 pinch   : a column that has become tiny or paper-thin ends (columns do
               terminate with depth): its short sides are swapped away until
               it is a small triangle, and the triangle shrinks to a point.
"""

from dataclasses import dataclass
from itertools import chain

import numpy as np

from physics import L, minimg, unwrap, energy, energy_gradient, cell_geometry


@dataclass
class Topology:
    """Flat 'half-edge' arrays so the physics can be vectorised."""
    he_cell: np.ndarray   # which cell the half-edge belongs to
    he_v: np.ndarray      # start junction
    he_vn: np.ndarray     # end junction (next vertex of the same cell)
    he_vp: np.ndarray     # previous vertex of the same cell (for corner angles)
    ref: np.ndarray       # a reference junction per cell (for unwrapping)
    ncell: int
    a0: float             # mean column area
    l0: float             # sqrt(a0), the column size


class CrackNetwork:
    T1_LENGTH = 0.08      # swap when a crack is shorter than this * l0
    T2_AREA = 0.10        # a column smaller than this * a0 pinches out
    PINCH_WIDTH = 0.12    # ... or one thinner than this * l0
    BEND_STRAIGHT = 12.0  # a bend point within this many degrees of straight is dropped
    PINCH_SWAP = 0.3      # a dying column swaps away sides shorter than this * l0

    def __init__(self, X, cells):
        self.X = np.asarray(X, float) % L
        self.cells = [list(c) for c in cells]
        self.cell_ids = list(range(len(self.cells)))   # stable names, kept through all changes
        self.n_T1 = self.n_T2 = 0
        self.rebuild()

    # ---------------------------------------------------------------- bookkeeping
    def rebuild(self):
        """Flat half-edge arrays from the cell lists (vectorised)."""
        cells = self.cells
        n = len(cells)
        lens = np.fromiter(map(len, cells), int, n)
        he_v = np.fromiter(chain.from_iterable(cells), int, int(lens.sum()))
        he_cell = np.repeat(np.arange(n), lens)
        start = np.cumsum(lens) - lens            # first half-edge of every cell
        s, m = start[he_cell], lens[he_cell]
        pos = np.arange(len(he_v)) - s            # position inside its cell
        a0 = L * L / n
        self.topo = Topology(he_cell, he_v, he_v[s + (pos + 1) % m], he_v[s + (pos - 1) % m],
                             he_v[start], n, a0, np.sqrt(a0))
        self._edge_owner = self._vcells = None    # rebuilt when first needed

    @property
    def edge_owner(self):
        """(a, b) -> the column that has the side a -> b."""
        if self._edge_owner is None:
            t = self.topo
            self._edge_owner = dict(zip(zip(t.he_v.tolist(), t.he_vn.tolist()),
                                        t.he_cell.tolist()))
        return self._edge_owner

    @property
    def vcells(self):
        """For every junction, the columns that meet there."""
        if self._vcells is None:
            self._vcells = [[] for _ in range(len(self.X))]
            for v, i in zip(self.topo.he_v.tolist(), self.topo.he_cell.tolist()):
                self._vcells[v].append(i)
        return self._vcells

    @property
    def ncell(self):
        return self.topo.ncell

    def energy(self, gamma):
        return energy(self.X, self.topo, gamma)

    def areas(self):
        p0 = minimg(self.X[self.topo.he_v] - self.X[self.topo.ref][self.topo.he_cell])
        p1 = minimg(self.X[self.topo.he_vn] - self.X[self.topo.ref][self.topo.he_cell])
        c = p0[:, 0] * p1[:, 1] - p1[:, 0] * p0[:, 1]
        return 0.5 * np.bincount(self.topo.he_cell, c, self.ncell)

    # ---------------------------------------------------------------- dynamics
    def relax_step(self, gamma, dt, max_move):
        """One depth increment: gradient step, then topology changes."""
        G = energy_gradient(self.X, self.topo, gamma)
        dX = -dt * self.topo.a0 * G
        biggest = np.hypot(*dX.T).max()
        cap = max_move * self.topo.l0
        if biggest > cap:
            dX *= cap / biggest
        self.X = (self.X + dX) % L
        self._touched = set()                 # junctions already swapped in this step
        self._T1_swaps()
        self._T2_pinch()
        self._straighten()

    # ---------------------------------------------------------------- bends
    # A network grown from curved cracks (crack_growth.py) also has "bend
    # points": vertices on a crack that are not junctions (only 2 columns meet
    # there).  The crack energy pulls them straight; once a bend is nearly
    # straight, or one of its two pieces is tiny, the point is dropped.
    def _straighten(self):
        t = self.topo
        deg = np.bincount(t.he_v, minlength=len(self.X))
        if not (deg == 2).any():
            return
        from measure import corner_angles
        ang = corner_angles(self)
        ell = np.hypot(*minimg(self.X[t.he_vn] - self.X[t.he_v]).T)
        short = ell < self.T1_LENGTH * t.l0
        drop = set(t.he_v[(deg[t.he_v] == 2) & (np.abs(ang - 180) < self.BEND_STRAIGHT)])
        drop |= set(t.he_v[short & (deg[t.he_v] == 2)]) | set(t.he_vn[short & (deg[t.he_vn] == 2)])
        if drop:
            self._dissolve(drop)

    def _dissolve(self, verts):
        """Remove bend points (vertices shared by only two columns)."""
        changed = False
        for v in verts:
            cs = self.vcells[v]
            if len(cs) == 2 and all(len(self.cells[c]) > 3 for c in cs):
                for c in cs:
                    self.cells[c].remove(v)
                self.vcells[v] = []
                changed = True
        if changed:
            self.rebuild()
        return changed

    # ---------------------------------------------------------------- T1
    def _T1_swaps(self):
        """Swap every crack segment that has shrunk below T1_LENGTH."""
        t = self.topo
        lmin = self.T1_LENGTH * t.l0
        ell = np.hypot(*minimg(self.X[t.he_vn] - self.X[t.he_v]).T)
        cand = np.where((ell < lmin) & (t.he_v < t.he_vn))[0]
        touched, changed = getattr(self, "_touched", set()), False
        for h in cand[np.argsort(ell[cand])]:
            a, b = int(t.he_v[h]), int(t.he_vn[h])
            if a in touched or b in touched:
                continue
            if self._T1(a, b):
                touched |= {a, b}
                changed = True
        if changed:
            self.rebuild()

    def _T1(self, a, b):
        """Neighbour swap on the crack a-b (see the sketch at the top)."""
        if (a, b) not in self.edge_owner or (b, a) not in self.edge_owner:
            return False
        P, Q = self.edge_owner[(a, b)], self.edge_owner[(b, a)]
        Rs, Ss = set(self.vcells[a]) - {P, Q}, set(self.vcells[b]) - {P, Q}
        if len(Rs) != 1 or len(Ss) != 1:
            return False
        R, S = Rs.pop(), Ss.pop()
        if R == S or len(self.cells[P]) <= 3 or len(self.cells[Q]) <= 3:
            return False
        for c in (P, Q, R, S):                            # forget their old edges
            cc = self.cells[c]
            for k in range(len(cc)):
                self.edge_owner.pop((cc[k], cc[(k + 1) % len(cc)]), None)
        # before: P ..a,b..   Q ..b,a..   R ..w,a,u..     S ..v,b,t..
        # after : P ..a..     Q ..b..     R ..w,b,a,u..   S ..v,a,b,t..
        self.cells[P].remove(b)
        self.cells[Q].remove(a)
        self.cells[R].insert(self.cells[R].index(a), b)
        self.cells[S].insert(self.cells[S].index(b), a)
        for v, old, new in ((b, P, R), (a, Q, S)):
            self.vcells[v] = [new if c == old else c for c in self.vcells[v]]
        d = minimg(self.X[b] - self.X[a])
        mid = self.X[a] + 0.5 * d
        normal = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-15)          # points into P
        half = 0.75 * self.T1_LENGTH * self.topo.l0             # new crack: 1.5 x threshold
        self.X[a] = (mid + half * normal) % L
        self.X[b] = (mid - half * normal) % L
        for c in (P, Q, R, S):                            # register the new edges
            cc = self.cells[c]
            for k in range(len(cc)):
                self.edge_owner[(cc[k], cc[(k + 1) % len(cc)])] = c
        self.n_T1 += 1
        return True

    # ---------------------------------------------------------------- T2
    def _bad_cells(self):
        """Columns that have become tiny (area), paper-thin (mean width) or
        twisted (outline crossing itself)."""
        A, J, Per = cell_geometry(self.X, self.topo)
        width = A / (0.5 * Per)
        # a column whose outline has crossed itself (J/A^2 below the disc's 1/2pi)
        # is a sliver that has been squeezed flat: treat it like a tiny one
        twisted = (A > 0) & (J < 0.155 * A ** 2)
        small = (A < self.T2_AREA * self.topo.a0) | twisted
        thin = width < self.PINCH_WIDTH * self.topo.l0
        return A, small, small | thin

    def _T2_pinch(self, max_events=10):
        """A column that has become tiny or paper-thin ends.  Its short sides
        are swapped away until it is a small triangle, which then shrinks to
        a point (columns terminate with depth, as in real flows)."""
        for _ in range(max_events):
            A, small, bad = self._bad_cells()
            if not bad.any():
                return
            done = False
            for i in np.flatnonzero(bad)[np.argsort(A[bad])]:
                c = self.cells[i]
                bends = [v for v in c if len(self.vcells[v]) == 2]
                if small[i] and bends and self._dissolve(bends):
                    done = True
                    break
                if len(c) == 3:
                    done = small[i] and self._remove_triangle(i)
                else:
                    P = unwrap(self.X[c])
                    ell = np.hypot(*(np.roll(P, -1, 0) - P).T)
                    touched = getattr(self, "_touched", set())
                    for k in np.argsort(ell):
                        if not small[i] and ell[k] > self.PINCH_SWAP * self.topo.l0:
                            break                      # a thin column swaps only short sides
                        a, b = c[k], c[(k + 1) % len(c)]
                        if a in touched or b in touched:
                            continue                   # one swap per junction per step
                        if self._T1(a, b):
                            touched |= {a, b}
                            done = True
                            break
                if done:
                    self.rebuild()
                    break
            if not done:
                return

    def _remove_triangle(self, i):
        c = self.cells[i]
        nbrs = {self.edge_owner.get((c[(k + 1) % 3], c[k])) for k in range(3)}
        if None in nbrs or len(nbrs) != 3 or any(len(self.cells[j]) < 4 for j in nbrs):
            return False
        a, b, cc = c
        self.X[a] = unwrap(self.X[c]).mean(0) % L
        for j in nbrs:                                   # merge b and cc into a
            new = [a if v in (b, cc) else v for v in self.cells[j]]
            self.cells[j] = [v for k, v in enumerate(new) if v != new[k - 1]]
        del self.cells[i]
        del self.cell_ids[i]
        self.n_T2 += 1
        return True
