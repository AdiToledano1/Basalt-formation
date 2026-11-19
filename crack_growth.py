"""
crack_growth.py - the cooling surface cracks, starting from intact crust.
=========================================================================

No seed pattern: at step 0 there are no cracks at all.  Three physical rules
then produce the surface network, and its T junctions, by themselves:

1) NUCLEATION where the crust is most stressed.
   Next to a crack the crust has relaxed; far from every crack it is still
   fully stretched.  So a new crack starts at the point farthest from all
   existing cracks (the best of a few random candidate points: rock has
   random flaws).  Cracks keep appearing as the crust cools, until there are
   as many as requested.

2) GROWTH perpendicular to the largest tension.
   A crack grows from both of its tips by a small step each time step.  Near
   an existing crack, the pull across that crack is relieved, so the largest
   pull is parallel to the old crack.  A crack runs across the largest pull,
   so a tip near an old crack turns toward the old crack's normal direction.
   The turn is stronger the closer the tip is:  weight  w = exp(-d / l_relax).

3) ARREST at an older crack.
   A tip that reaches a crack stops there: nothing can be pulled across the
   gap.  Because of rule 2 it arrives at nearly 90 degrees: a T junction.

When every tip has stopped, the crack lines are turned into the column
network used by the rest of the program (to_network).
"""

import numpy as np
from scipy.spatial import cKDTree

from physics import L, minimg
from network import CrackNetwork


class GrowthError(RuntimeError):
    pass


class Crack:
    """One crack: two arms growing in opposite directions from where it started."""

    def __init__(self, p, direction, step):
        p = np.array(p, float)
        self.arms = [[p.copy()], [p.copy()]]       # unwrapped coordinates
        self.ids = [[], []]                        # global index of each arm point
        self.dirs = [np.array(direction, float), -np.array(direction, float)]
        self.active = [True, True]
        self.end = [None, None]       # (host crack, host arm, host segment k, u) once stopped
        self.born = step


class CrackGrowth:
    def __init__(self, n_cracks, rng, speed=0.06, influence=0.8, relax=0.3, turn=0.5,
                 nucleation_steps=45, candidates=48):
        self.N = n_cracks
        self.rng = rng
        self.l0 = L / np.sqrt(n_cracks)          # final column width
        self.ds = speed * self.l0                # how far a tip moves per step
        self.R = influence * self.l0             # tips feel cracks closer than this
        self.l_relax = relax * self.l0           # decay length of the stress relief
        self.turn = turn
        self.T_nuc = nucleation_steps
        self.n_cand = candidates
        self.cracks = []
        self.step_no = 0
        self.meet_angles = []                    # angle at which each tip met its host
        self.junction_pts = []
        self.history = []
        # every crack point, for fast neighbour searches
        self._n = 0                              # number of points stored
        self._Pbuf = np.empty((4096, 2))         # point positions
        self._Ibuf = np.empty((4096, 5), int)    # crack, arm, index, prev, next

    # ------------------------------------------------------------------ status
    @property
    def n_junctions(self):
        return len(self.junction_pts)

    @property
    def active_tips(self):
        return [(i, a) for i, c in enumerate(self.cracks) for a in (0, 1) if c.active[a]]

    @property
    def done(self):
        return len(self.cracks) >= self.N and not self.active_tips

    # ------------------------------------------------------------------ bookkeeping
    def _new_point(self, ci, a, first=False):
        """Register the last point of arm a of crack ci in the search arrays."""
        c = self.cracks[ci]
        g = self._n
        if g == len(self._Pbuf):                 # full: double the storage
            self._Pbuf = np.concatenate([self._Pbuf, np.empty_like(self._Pbuf)])
            self._Ibuf = np.concatenate([self._Ibuf, np.empty_like(self._Ibuf)])
        w = c.arms[a][-1] % L
        w[w >= L] -= L
        if first:
            prev = -1
            c.ids[0].append(g)
            c.ids[1].append(g)
        else:
            prev = c.ids[a][-1]
            if self._Ibuf[prev, 4] == -1:
                self._Ibuf[prev, 4] = g
            c.ids[a].append(g)
        self._Pbuf[g] = w
        self._Ibuf[g] = (ci, a, len(c.arms[a]) - 1, prev, -1)
        self._n = g + 1

    def _arrays(self):
        n = self._n
        P, Ib = self._Pbuf[:n].copy(), self._Ibuf[:n].copy()   # snapshot of this step
        return cKDTree(P, boxsize=L), P, Ib[:, :3], Ib[:, 3], Ib[:, 4]

    # ------------------------------------------------------------------ one step
    def step(self):
        if self.step_no > 3000:
            raise GrowthError("crack growth did not finish")
        self._nucleate()
        self._grow()
        self.step_no += 1
        self.history.append((self.step_no, len(self.cracks), self.n_junctions,
                             len(self.active_tips)))

    def run(self):
        while not self.done:
            self.step()
        return self

    # ------------------------------------------------------------------ rule 1
    def _nucleate(self):
        due = min(self.N, int(np.ceil(self.N * min(1.0, (self.step_no + 1) / self.T_nuc))))
        if len(self.cracks) >= due:
            return
        tree, P = self._arrays()[:2] if self._n else (None, None)
        new = []
        while len(self.cracks) < due:
            cand = self.rng.random((self.n_cand, 2)) * L
            if tree is not None:
                d, idx = tree.query(cand)
            else:
                d, idx = np.full(self.n_cand, np.inf), np.zeros(self.n_cand, int)
            for q in new:                         # cracks started in this same step
                d = np.minimum(d, np.hypot(*minimg(cand - q).T))
            j = int(np.argmax(d))
            if d[j] < 0.15 * self.l0:
                break                             # no room left anywhere
            p = cand[j]
            if tree is not None and d[j] < self.R:
                n = minimg(P[idx[j]] - p)         # rule 2 already at birth:
                direction = n / np.hypot(*n)      # run along the nearest crack's normal
            else:
                th = self.rng.uniform(0, np.pi)
                direction = np.array([np.cos(th), np.sin(th)])
            self.cracks.append(Crack(p, direction, self.step_no))
            self._new_point(len(self.cracks) - 1, 0, first=True)
            new.append(p)

    # ------------------------------------------------------------------ rules 2 + 3
    def _arc(self, ci, a, a2, k2):
        """Number of points along crack ci from the tip of arm a to point (a2, k2)."""
        n = len(self.cracks[ci].arms[a])
        return (n - 1 - k2) if a2 == a else (n - 1 + k2)

    def _grow(self):
        tips = self.active_tips
        if not tips:
            return
        tree, P, meta, prev, nxt = self._arrays()
        tip_w = np.array([self.cracks[i].arms[a][-1] for i, a in tips]) % L
        tip_w[tip_w >= L] -= L
        k = min(16, len(P))
        dist, idx = tree.query(tip_w, k=k, distance_upper_bound=self.R)
        dist, idx = dist.reshape(len(tips), -1), idx.reshape(len(tips), -1)
        near = tree.query_ball_point(tip_w, 3.5 * self.ds)
        has_prev = prev >= 0
        P1 = np.where(has_prev[:, None], P[np.maximum(prev, 0)], P)
        P2 = P1 + minimg(P - P1)                  # segment ending at each point
        S = (P1, P2, meta[:, 0], meta[:, 1], meta[:, 2], prev, nxt)
        # segments added during this step (filled row by row, so no list -> array
        # conversion is needed for every tip)
        fresh = [np.empty((len(tips), 2)), np.empty((len(tips), 2)),
                 np.empty((len(tips), 3), int), 0]
        for t in self.rng.permutation(len(tips)):
            ci, a = tips[t]
            c = self.cracks[ci]
            p, pw, d = c.arms[a][-1], tip_w[t], c.dirs[a]
            # --- rule 2: turn toward the normal of the nearest other crack
            for dd, ii in zip(dist[t], idx[t]):
                if not np.isfinite(dd):
                    break
                c2, a2, k2 = meta[ii]
                if c2 == ci and self._arc(ci, a, a2, k2) * self.ds < 2 * self.R:
                    continue                      # ignore the tip's own recent path
                n = minimg(P[ii] - pw)
                nn = np.hypot(*n)
                if nn < 1e-12:
                    break
                n /= nn
                target = n if d @ n >= 0 else -n
                d = d + self.turn * np.exp(-nn / self.l_relax) * (target - d)
                d /= np.hypot(*d)
                self._sensed = c2
                break
            c.dirs[a] = d
            step = self.ds * d
            # --- rule 3: stop at the first crack the step would cross
            hit = self._first_hit(ci, a, pw, step, near[t], S, fresh)
            move = step if hit is None else hit[0]
            c.arms[a].append(p + move)
            self._new_point(ci, a)
            nf = fresh[3]
            fresh[0][nf], fresh[1][nf] = pw, pw + move
            fresh[2][nf] = (ci, a, len(c.arms[a]) - 2)
            fresh[3] = nf + 1
            if hit is not None:
                _, host, tangent = hit
                c.active[a] = False
                c.end[a] = host
                xw = (p + move) % L
                self.junction_pts.append(xw)
                cosang = abs(move / max(np.hypot(*move), 1e-15) @ tangent)
                self.meet_angles.append(np.degrees(np.arccos(min(1.0, cosang))))

    def _first_hit(self, ci, a, pw, step, near, S, fresh):
        """First crack segment crossed by the step pw -> pw + step (or None).
        Returns (hit point relative to pw, (host crack, arm, k, u), host direction)."""
        P1, P2, mc, ma, mk, prev, nxt = S
        near = np.asarray(near, int)
        ids = np.concatenate([near[prev[near] >= 0], nxt[near][nxt[near] >= 0]])
        q1 = pw + minimg(P1[ids] - pw)
        q2 = q1 + (P2[ids] - P1[ids])
        c2, a2, k2 = mc[ids], ma[ids], mk[ids] - 1          # segment k2 -> k2 + 1
        nf = fresh[3]
        if nf:
            F1, F2, fm = fresh[0][:nf], fresh[1][:nf], fresh[2][:nf]
            fq1 = pw + minimg(F1 - pw)
            q1 = np.vstack([q1, fq1])
            q2 = np.vstack([q2, fq1 + (F2 - F1)])
            c2 = np.concatenate([c2, fm[:, 0]])
            a2 = np.concatenate([a2, fm[:, 1]])
            k2 = np.concatenate([k2, fm[:, 2]])
        if len(q1) == 0:
            return None
        n_own = len(self.cracks[ci].arms[a])
        arc = np.where(a2 == a, n_own - 2 - k2, n_own + k2)   # points from the tip
        ok = ~((c2 == ci) & (arc < 3))
        s_ = q2 - q1
        den = step[0] * s_[:, 1] - step[1] * s_[:, 0]
        ok &= np.abs(den) > 1e-18
        den = np.where(ok, den, 1.0)
        w = q1 - pw
        t = (w[:, 0] * s_[:, 1] - w[:, 1] * s_[:, 0]) / den
        u = (w[:, 0] * step[1] - w[:, 1] * step[0]) / den
        ok &= (t > 1e-9) & (t <= 1) & (u >= 0) & (u <= 1)
        if not ok.any():
            return None
        j = np.flatnonzero(ok)[np.argmin(t[ok])]
        c2j, a2j, kj, uj = int(c2[j]), int(a2[j]), int(k2[j]), float(u[j])
        host = self.cracks[c2j]
        seglen = np.hypot(*s_[j])
        if kj + 1 == len(host.arms[a2j]) - 1 and seglen > 0:
            # keep clear of the host's tip (live) or of its end junction (stopped)
            uj = min(uj, max(0.0, 1 - 1e-3 * self.ds / seglen))
        X = q1[j] + uj * s_[j]
        return X - pw, (c2j, a2j, kj, uj), s_[j] / max(seglen, 1e-15)

    # ------------------------------------------------------------------ to columns
    def to_network(self):
        """Turn the finished crack lines into the column network.

        Junctions are the places where a tip stopped.  Along every crack, the
        junctions sitting on it cut it into crack segments.  Each column is
        traced by walking along segments, always taking the next segment
        clockwise at each junction, so that the column stays on the left."""
        if not self.done:
            raise GrowthError("cracks are still growing")
        C = self.cracks
        polys, svals, offs = [], [], []
        for c in C:
            A0, A1 = np.array(c.arms[0]), np.array(c.arms[1])
            poly = np.vstack([A0[::-1], A1[1:]])
            polys.append(poly)
            svals.append(np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]))
            offs.append(len(A0) - 1)

        def pidx(ci, arm, k):                    # arm point -> index along the polyline
            return offs[ci] - k if arm == 0 else offs[ci] + k

        def pos(ci, s):
            return np.array([np.interp(s, svals[ci], polys[ci][:, 0]),
                             np.interp(s, svals[ci], polys[ci][:, 1])])

        def direction(ci, s, forward):
            sv = svals[ci]
            i = int(np.clip(np.searchsorted(sv, s, side="right" if forward else "left") - 1,
                            0, len(sv) - 2))
            v = polys[ci][i + 1] - polys[ci][i]
            v = v / max(np.hypot(*v), 1e-15)
            return v if forward else -v

        # junction nodes: node 2*ci + a is where arm a of crack ci stopped
        on = [[] for _ in C]
        X = np.zeros((2 * len(C), 2))
        for ci, c in enumerate(C):
            for a in (0, 1):
                node = 2 * ci + a
                X[node] = c.arms[a][-1] % L
                on[ci].append((0.0 if a == 0 else svals[ci][-1], node))
                h, ha, hk, hu = c.end[a]
                i1, i2 = pidx(h, ha, hk), pidx(h, ha, hk + 1)
                on[h].append((svals[h][i1] + hu * (svals[h][i2] - svals[h][i1]), node))
        X[X >= L] -= L

        # crack segments between neighbouring junctions along each crack, kept as
        # simplified polylines so that curved cracks keep their shape
        edges = {}                # id -> [u, v, polyline relative to u, angle at u, angle at v]
        for ci in range(len(C)):
            nodes = sorted(on[ci])
            sv = svals[ci]
            for (sa, na), (sb, nb) in zip(nodes[:-1], nodes[1:]):
                inner = polys[ci][(sv > sa) & (sv < sb)]
                line = np.vstack([pos(ci, sa), inner, pos(ci, sb)])
                keep = np.r_[True, np.hypot(*np.diff(line, axis=0).T) > 1e-3 * self.ds]
                keep[-1] = True
                line = line[keep]
                ta, tb = direction(ci, sa, True), direction(ci, sb, False)
                edges[len(edges)] = [na, nb, line - line[0],
                                     np.arctan2(ta[1], ta[0]), np.arctan2(tb[1], tb[0])]
        self._tidy(edges)
        self._simplify_edges(edges, X)

        # trace the columns; a crack segment with the same column on both sides
        # (a "bridge" to a closed island of cracks) is removed, and we trace again
        while True:
            out, faces = self._trace(edges)
            face_of = {}
            for f, hs in enumerate(faces):
                for e, sgn in hs:
                    face_of[(e, sgn)] = f
            bridges = [e for e in edges if face_of[(e, 1)] == face_of[(e, -1)]]
            if not bridges:
                break
            for e in bridges:
                del edges[e]
            self._tidy(edges)

        # vertices: the junctions, plus the bend points along each crack segment
        keep = sorted(out)
        vid = {u: i for i, u in enumerate(keep)}
        pts = [X[u] for u in keep]
        bends = {}
        for e, (u, v, rel, _, _) in edges.items():
            bends[e] = list(range(len(pts), len(pts) + len(rel) - 2))
            pts.extend((X[u] + rel[1:-1]) % L)
        pts = np.array(pts)
        pts[pts >= L] -= L
        cells = []
        for hs in faces:
            cell, path = [], [np.zeros(2)]
            for e, sgn in hs:
                u, v, rel, _, _ = edges[e]
                cell.append(vid[u if sgn > 0 else v])
                cell.extend(bends[e] if sgn > 0 else bends[e][::-1])
                path.append(path[-1] + sgn * rel[-1])
            path = np.array(path)
            if np.hypot(*path[-1]) > 1e-6:
                raise GrowthError("a column wraps around the patch")
            if np.ptp(path, axis=0).max() > 0.45 * L:
                raise GrowthError("a column is too large for the periodic patch")
            cells.append(cell)

        net = CrackNetwork(pts, cells)
        net._T2_pinch(max_events=10 * net.ncell)      # remove any sliver left over
        net.n_T1 = net.n_T2 = 0
        A = net.areas()
        if A.min() <= 0 or abs(A.sum() - L * L) > 1e-6:
            raise GrowthError("could not build valid columns from the cracks")
        return net

    @staticmethod
    def _trace(edges):
        """Walk around every column: arriving at a junction, leave by the next
        crack segment clockwise, so the column is always on the left.
        Returns the outgoing half-edges per junction and, for each column,
        its list of (segment id, direction)."""
        out = {}
        for e, (u, v, rel, au, av) in edges.items():
            out.setdefault(u, []).append((au, e, +1))
            out.setdefault(v, []).append((av, e, -1))
        if any(len(o) != 3 for o in out.values()):
            raise GrowthError("a junction does not join exactly three cracks")
        for o in out.values():
            o.sort(key=lambda h: h[0])
        used, faces = set(), []
        for u0 in out:
            for j0 in range(3):
                if (u0, j0) in used:
                    continue
                hs, u, j = [], u0, j0
                while (u, j) not in used:
                    used.add((u, j))
                    _, e, sgn = out[u][j]
                    hs.append((e, sgn))
                    eu, ev = edges[e][0], edges[e][1]
                    v = ev if sgn > 0 else eu
                    back = next(m for m, h in enumerate(out[v]) if h[1] == e and h[2] == -sgn)
                    u, j = v, (back - 1) % 3
                faces.append(hs)
        return out, faces

    def _simplify_edges(self, edges, X, tol=0.02, rounds=8):
        """Replace each crack segment's many points by a few bend points
        (Douglas-Peucker), without letting two simplified cracks cross:
        wherever they would, those two are simplified less."""
        raw = {e: rec[2] for e, rec in edges.items()}
        tols = {e: tol * self.l0 for e in edges}
        for _ in range(rounds):
            for e in edges:
                edges[e][2] = _simplify(raw[e], tols[e]) if tols[e] > 0 else raw[e]
            bad = _crossing_edges(edges, X)
            if not bad:
                return
            for e in bad:
                tols[e] = tols[e] / 3 if tols[e] > 1e-4 * self.l0 else 0.0
        for e in edges:
            if tols[e] < tol * self.l0:
                edges[e][2] = raw[e]

    @staticmethod
    def _tidy(edges):
        """Clean up the few places where the cracks do not form proper
        three-way junctions (two cracks that end on each other enclose a
        sliver bounded by only two segments, or a crack closes on itself):
          - a loop, or one of two parallel segments, is deleted;
          - a junction left with one crack is deleted with that crack;
          - a junction left with two cracks is dissolved (they are joined)."""
        while True:
            inc = {}
            for e, (u, v, *_) in edges.items():
                inc.setdefault(u, []).append(e)
                inc.setdefault(v, []).append(e)
            loop = next((e for e, (u, v, *_) in edges.items() if u == v), None)
            if loop is not None:
                del edges[loop]
                continue
            seen, dup = {}, None
            for e, (u, v, *_) in edges.items():
                key = (min(u, v), max(u, v))
                if key in seen:
                    dup = e
                    break
                seen[key] = e
            if dup is not None:
                del edges[dup]
                continue
            n1 = next((n for n, es in inc.items() if len(es) == 1), None)
            if n1 is not None:
                del edges[inc[n1][0]]
                continue
            n2 = next((n for n, es in inc.items() if len(es) == 2), None)
            if n2 is None:
                return
            ea, eb = inc[n2]

            def toward(e):          # (far node, polyline far -> n2 relative to far, angle at far)
                u, v, rel, au, av = edges[e]
                return (u, rel, au) if v == n2 else (v, rel[::-1] - rel[-1], av)
            fa, ra, aa = toward(ea)
            fb, rb, ab = toward(eb)
            back = rb[::-1] - rb[-1]                      # n2 -> fb, relative to n2
            del edges[ea], edges[eb]
            edges[max(list(edges) + [ea, eb]) + 1] = [fa, fb, np.vstack([ra, ra[-1] + back[1:]]),
                                                     aa, ab]


def _simplify(P, tol):
    """Douglas-Peucker: drop points that lie within tol of the straight line."""
    if len(P) <= 2:
        return P
    a, b = P[0], P[-1]
    ab = b - a
    L2 = ab @ ab
    t = np.clip((P - a) @ ab / L2, 0, 1) if L2 > 0 else np.zeros(len(P))
    dist = np.hypot(*(P - (a + t[:, None] * ab)).T)
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return np.vstack([a, b])
    left, right = _simplify(P[:i + 1], tol), _simplify(P[i:], tol)
    return np.vstack([left[:-1], right])


def _crossing_edges(edges, X):
    """Ids of crack segments (polylines) that cross another one."""
    A, B, owner, index = [], [], [], []
    for e, (u, v, rel, *_) in edges.items():
        P = X[u] + rel
        A.append(P[:-1]); B.append(P[1:])
        owner += [e] * (len(P) - 1)
        index += range(len(P) - 1)
    A, B = np.vstack(A), np.vstack(B)
    owner, index = np.array(owner), np.array(index)
    mid = ((A + B) / 2) % L
    mid[mid >= L] -= L
    reach = np.hypot(*(B - A).T).max()
    pairs = cKDTree(mid, boxsize=L).query_pairs(reach, output_type="ndarray")
    if len(pairs) == 0:
        return set()
    i, j = pairs[:, 0], pairs[:, 1]
    keep = ~((owner[i] == owner[j]) & (np.abs(index[i] - index[j]) <= 1))
    i, j = i[keep], j[keep]
    p, r = A[i], B[i] - A[i]
    q = p + minimg(A[j] - p)
    s = B[j] - A[j]
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    ok = np.abs(den) > 1e-18
    den = np.where(ok, den, 1.0)
    w = q - p
    t = (w[:, 0] * s[:, 1] - w[:, 1] * s[:, 0]) / den
    u = (w[:, 0] * r[:, 1] - w[:, 1] * r[:, 0]) / den
    hit = ok & (t > 1e-6) & (t < 1 - 1e-6) & (u > 1e-6) & (u < 1 - 1e-6)
    return set(owner[i[hit]]) | set(owner[j[hit]])
