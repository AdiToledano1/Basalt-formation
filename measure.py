"""
measure.py - what we MEASURE on the network.  Hexagons are detected here,
never imposed anywhere else.
=========================================================================

Junction angles
    At every junction three columns meet; their corner angles add to 360.
    T junction : 90 / 90 / 180      Y junction : 120 / 120 / 120
    We track the LARGEST of the three: 180 for a T, 120 for a perfect Y.

Corners vs junctions of a column
    topological sides = number of junctions on the column's boundary.
        By Euler's formula (V - E + F = 0 on a torus, 3 cracks per junction)
        the average is exactly 6, from the very first step.
    geometric sides  = number of REAL corners, i.e. junctions where the
        column's boundary actually bends (angle < CORNER_CUT).  The flat side
        of a T junction (180 deg) is not a corner.  This is what your eye
        counts, and what "hexagon" means in the plots.
"""

import numpy as np

from physics import minimg, cell_geometry

CORNER_CUT = 155.0     # degrees; flatter than this is a point on a straight crack
T_LIKE = 150.0         # a junction whose largest angle exceeds this looks like a T


def corner_angles(net):
    """Interior angle of every column at every one of its junctions
    (one value per half-edge, at junction he_v of column he_cell)."""
    t, X = net.topo, net.X
    a = minimg(X[t.he_vp] - X[t.he_v])
    b = minimg(X[t.he_vn] - X[t.he_v])
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return np.degrees(np.arctan2(-cross, (a * b).sum(1))) % 360


def measure(net, gamma=0.0):
    t = net.topo
    ang = corner_angles(net)
    nv = len(net.X)
    count = np.bincount(t.he_v, minlength=nv)
    largest = np.full(nv, -np.inf)
    np.maximum.at(largest, t.he_v, ang)
    largest = largest[count == 3]

    at_junction = count[t.he_v] == 3           # (bend points on a crack are not junctions)
    corners = np.bincount(t.he_cell, (ang < CORNER_CUT) & at_junction, t.ncell).astype(int)
    junctions = np.bincount(t.he_cell, at_junction, t.ncell).astype(int)
    A, J, P = cell_geometry(net.X, t)
    E_el, E_cr = net.energy(gamma)
    return dict(
        angles=ang[count[t.he_v] == 3],          # all corner angles at junctions
        largest_angle=largest,                   # one per junction
        corners=corners,                         # geometric sides, per column
        junctions=junctions,                     # topological sides, per column
        shape_J=J / A ** 2,                      # elastic-energy shape factor
        shape_P=P / np.sqrt(np.maximum(A, 1e-300)),   # crack length per column
        energy=(E_el + E_cr) / t.ncell,
    )


def summary(net, gamma, step):
    """Small dict of scalars for the history curves."""
    m = measure(net, gamma)
    c = np.clip(m["corners"], 3, 8)
    return dict(
        step=step,
        largest_angle=m["largest_angle"].mean(),
        frac_T=(m["largest_angle"] > T_LIKE).mean(),
        corners_mean=m["corners"].mean(),
        junctions_mean=m["junctions"].mean(),
        frac={k: (c == k).mean() for k in range(3, 9)},
        shape_J=m["shape_J"].mean(),
        shape_P=m["shape_P"].mean(),
        energy=m["energy"],
        ncell=net.ncell,
        n_T1=net.n_T1,
        gamma=gamma,
    )
