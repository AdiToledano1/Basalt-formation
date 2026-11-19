"""
physics.py - the energy of a crack network and its exact gradient.
====================================================================

This is the only file with physics in it.  It knows nothing about hexagons.

A horizontal slice through the cooling lava is cut by cracks into polygons
(column cross-sections).  The slice has two kinds of energy:

1) ELASTIC (shear-lag) energy
   The new layer wants to shrink by a thermal strain eps, but it is glued to
   the hotter rock below by an interface of stiffness k.  A column isolated by
   cracks shrinks freely toward some point c, so its displacement is
   u(x) = eps * (x - c) and the stored energy is

        E_cell = (k eps^2 / 2) * integral_cell |x - c|^2 dA

   Minimising over c puts c at the centroid, so  E_cell  is proportional to
   J = the polar second moment of area of the cell about its centroid.

2) CRACK (Griffith) energy
   Opening new crack surface costs Gamma per unit area, so per unit depth

        E_crack = Gamma * (total crack length)

   Dimensionless total energy (a0 = mean column area, l0 = sqrt(a0)):

        E = sum_cells J_i / a0^2  +  gamma * sum_cracks l_e / l0

POLYGON FORMULAS  (vertices p_k = (x_k, y_k), counter-clockwise, k+1 cyclic)
----------------------------------------------------------------------------
   c_k = x_k y_{k+1} - x_{k+1} y_k                    (twice a triangle's area)
   A   = 1/2  sum c_k                                  (shoelace formula)
   S   = 1/6  sum (p_k + p_{k+1}) c_k                  (first moment = A * centroid)
   Q   = 1/12 sum (x_k^2 + x_k x_{k+1} + x_{k+1}^2
                 + y_k^2 + y_k y_{k+1} + y_{k+1}^2) c_k (second moment about origin)
   J   = Q - |S|^2 / A                                 (parallel-axis theorem)

GRADIENT  (needed for dX/dz = -dE/dX; derived by hand, checked numerically)
----------------------------------------------------------------------------
   dJ = dQ - 2 S.dS / A + |S|^2 dA / A^2
   and each sum is a sum of edge terms f(p_k, p_{k+1}), so we differentiate
   every edge term with respect to its two end points, e.g.
   d c_k / d(x_k, y_k, x_{k+1}, y_{k+1}) = ( y_{k+1}, -x_{k+1}, -y_k, x_k ).
   d(length)/d(p_{k+1}) = unit vector along the edge.

Reference values for the dashed lines on the plots (pure geometry, the
simulation never uses them):
   J/A^2       triangle 0.1925   square 0.1667   hexagon 0.1604
   P/sqrt(A)   triangle 4.559    square 4.000    hexagon 3.722
"""

import numpy as np

L = 1.0          # side of the periodic box (a patch from the middle of a big flow)
_TINY = 1e-12

REFERENCE = {    # (J/A^2, P/sqrt(A)) for regular tilings - plotting only
    "square": (1 / 6, 4.0),
    "hexagon": (5 * np.sqrt(3) / 54, 2 * np.sqrt(2 * np.sqrt(3))),
}


# ----------------------------------------------------------------------------
# Periodic-box helpers
# ----------------------------------------------------------------------------
def minimg(d):
    """Minimum-image vector on the periodic box."""
    return d - L * np.rint(d / L)


def unwrap(P):
    """Make a polygon contiguous (it may straddle the box edge)."""
    return P[0] + minimg(P - P[0])


# ----------------------------------------------------------------------------
# Single polygon (used when building initial patterns)
# ----------------------------------------------------------------------------
def polygon_moments(P):
    """Area, centroid, J, and the 2x2 covariance (for the long axis)."""
    P = unwrap(np.asarray(P, float))
    x, y = P[:, 0] - P[0, 0], P[:, 1] - P[0, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    c = x * yn - xn * y
    A = 0.5 * c.sum()
    Sx, Sy = ((x + xn) * c).sum() / 6, ((y + yn) * c).sum() / 6
    Ixx = ((x * x + x * xn + xn * xn) * c).sum() / 12          # integral x^2
    Iyy = ((y * y + y * yn + yn * yn) * c).sum() / 12          # integral y^2
    Ixy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * c).sum() / 24
    cx, cy = Sx / A, Sy / A
    C = np.array([[Ixx - A * cx * cx, Ixy - A * cx * cy],
                  [Ixy - A * cx * cy, Iyy - A * cy * cy]])
    return A, P[0] + [cx, cy], C[0, 0] + C[1, 1], C


# ----------------------------------------------------------------------------
# Whole network, vectorised over half-edges
# ----------------------------------------------------------------------------
def _edge_coords(X, topo):
    """End points of every half-edge, relative to its cell's first vertex."""
    R = X[topo.ref][topo.he_cell]
    p0 = minimg(X[topo.he_v] - R)
    p1 = minimg(X[topo.he_vn] - R)
    return p0, p1


def cell_geometry(X, topo):
    """Per-cell area A, polar moment J and perimeter P (all vectorised)."""
    p0, p1 = _edge_coords(X, topo)
    x0, y0, x1, y1 = p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1]
    hc, n = topo.he_cell, topo.ncell
    c = x0 * y1 - x1 * y0
    s = x0 * x0 + x0 * x1 + x1 * x1 + y0 * y0 + y0 * y1 + y1 * y1
    A = 0.5 * np.bincount(hc, c, n)
    Sx = np.bincount(hc, (x0 + x1) * c, n) / 6
    Sy = np.bincount(hc, (y0 + y1) * c, n) / 6
    Q = np.bincount(hc, s * c, n) / 12
    with np.errstate(divide="ignore", invalid="ignore"):   # a collapsing cell may have A = 0
        J = Q - (Sx * Sx + Sy * Sy) / A
    ell = np.hypot(*(p1 - p0).T)
    P = np.bincount(hc, ell, n)
    return A, J, P


def energy(X, topo, gamma):
    """(E_elastic, E_crack), dimensionless."""
    A, J, P = cell_geometry(X, topo)
    return J.sum() / topo.a0 ** 2, gamma * 0.5 * P.sum() / topo.l0   # each crack has 2 sides


def energy_gradient(X, topo, gamma):
    """dE/dX for every junction, from the formulas in the module docstring."""
    p0, p1 = _edge_coords(X, topo)
    x0, y0, x1, y1 = p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1]
    hc, n = topo.he_cell, topo.ncell
    c = x0 * y1 - x1 * y0
    s = x0 * x0 + x0 * x1 + x1 * x1 + y0 * y0 + y0 * y1 + y1 * y1
    A = 0.5 * np.bincount(hc, c, n)
    Sx = np.bincount(hc, (x0 + x1) * c, n) / 6
    Sy = np.bincount(hc, (y0 + y1) * c, n) / 6

    # derivative of each edge term with respect to (x0, y0, x1, y1)
    gA = 0.5 * np.stack([y1, -x1, -y0, x0], 1)
    gSx = np.stack([c + (x0 + x1) * y1, -(x0 + x1) * x1,
                    c - (x0 + x1) * y0, (x0 + x1) * x0], 1) / 6
    gSy = np.stack([(y0 + y1) * y1, c - (y0 + y1) * x1,
                    -(y0 + y1) * y0, c + (y0 + y1) * x0], 1) / 6
    gQ = np.stack([(2 * x0 + x1) * c + s * y1, (2 * y0 + y1) * c - s * x1,
                   (x0 + 2 * x1) * c - s * y0, (y0 + 2 * y1) * c + s * x0], 1) / 12

    # chain rule: dJ = dQ - 2 S.dS / A + |S|^2 dA / A^2
    Ah, Sxh, Syh = A[hc, None], Sx[hc, None], Sy[hc, None]
    gJ = gQ - 2 * (Sxh * gSx + Syh * gSy) / Ah + (Sxh ** 2 + Syh ** 2) / Ah ** 2 * gA
    gJ /= topo.a0 ** 2

    # crack length: d|p1 - p0| = unit vector
    d = p1 - p0
    u = d / np.maximum(np.hypot(*d.T), _TINY)[:, None]
    gL = gamma * 0.5 / topo.l0 * np.hstack([-u, u])

    g = gJ + gL
    G = np.zeros_like(X)
    np.add.at(G, topo.he_v, g[:, :2])
    np.add.at(G, topo.he_vn, g[:, 2:])
    return G


def gradient_check(X, topo, gamma, n_probe=40, h=1e-6, seed=0):
    """Largest relative error between the analytic gradient and
    central finite differences  (E(x+h) - E(x-h)) / 2h."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    G = energy_gradient(X, topo, gamma)
    used = np.unique(topo.he_v)
    worst = 0.0
    for v in rng.choice(used, min(n_probe, len(used)), replace=False):
        for d in (0, 1):
            x0 = X[v, d]
            X[v, d] = x0 + h; Ep = sum(energy(X, topo, gamma))
            X[v, d] = x0 - h; Em = sum(energy(X, topo, gamma))
            X[v, d] = x0
            fd = (Ep - Em) / (2 * h)
            worst = max(worst, abs(fd - G[v, d]) / max(abs(fd), abs(G[v, d]), 1e-3))
    return worst
