"""
simulation.py - run the crack front downward, step by step.
===========================================================

Two phases:

  forming   (only for the default "growth" start) the cooling surface cracks:
            cracks appear, grow and stop on older cracks, making T junctions
            (crack_growth.py).  Each step = a moment of time at the surface.
  deepening the crack front advances downward.  Depth z plays the role of
            time: each step = one small advance into the cooling lava.  Every
            junction moves a little down the energy gradient (physics.py,
            network.py), plus any T1 / T2 topology changes.

    sim = Simulation(Params(pattern="growth", n_cells=300, seed=3))
    sim.finish_forming()      # (or keep calling sim.advance() to watch it)
    sim.advance(100)          # 100 depth steps
    sim.history[-1]           # measured numbers (see measure.summary)
"""

from dataclasses import dataclass, asdict


from measure import summary
from patterns import make_pattern, make_growth, grown_network
from physics import L, cell_geometry, gradient_check


@dataclass
class Params:
    pattern: str = "growth"     # starting pattern: growth | cracking | voronoi | brick
    n_cells: int = 300          # number of columns in the patch
    seed: int = 3               # random seed for the starting pattern
    gamma: float = 0.1          # crack-energy weight (Griffith); keep >= ~0.05
    dt: float = 0.05            # step size of the gradient flow (mobility)
    max_move: float = 0.03      # largest junction move per step, in column sizes
    steps: int = 1500           # how deep to go


class InvertedCellError(RuntimeError):
    pass


class Simulation:
    FOLD_PATIENCE = 10          # a twisted column must straighten out within this many steps

    def __init__(self, params: Params, watch_forming=True):
        self.p = params
        self.step_no = 0
        self.history = []
        self.growth = None
        self.net = None
        self._bad_for = 0
        if params.pattern == "growth":
            self.growth = make_growth(params.n_cells, params.seed)
            if not watch_forming:
                self.finish_forming()
        else:
            self._start(make_pattern(params.pattern, params.n_cells, params.seed))

    def _start(self, net):
        self.net = net
        self.history = [summary(net, self.p.gamma, 0)]

    # ------------------------------------------------------------------ phases
    @property
    def forming(self):
        return self.net is None

    def finish_forming(self):
        if self.forming:
            self._start(grown_network(self.growth, self.p.n_cells, self.p.seed))

    @property
    def done(self):
        return not self.forming and self.step_no >= self.p.steps

    def gradient_error(self):
        self.finish_forming()
        return gradient_check(self.net.X, self.net.topo, self.p.gamma)

    # ------------------------------------------------------------------ stepping
    def advance(self, n=1, record_every=1):
        """Advance n steps: surface-cracking steps while forming, then depth steps."""
        if self.forming:
            for _ in range(n):
                self.growth.step()
                if self.growth.done:
                    self.finish_forming()
                    return
            return
        for _ in range(n):
            if self.done:
                break
            self.net.relax_step(self.p.gamma, self.p.dt, self.p.max_move)
            self.step_no += 1
            self._check()
            if self.step_no % record_every == 0 or self.done:
                self.history.append(summary(self.net, self.p.gamma, self.step_no))
        if self.history[-1]["step"] != self.step_no:
            self.history.append(summary(self.net, self.p.gamma, self.step_no))

    def _check(self):
        """While the network rewires, a column may fold over or twist for a step
        or two.  It must straighten out again quickly, and the columns must
        always tile the patch exactly.  (A twisted polygon is recognisable
        because J/A^2 < 1/(2 pi), which no real shape can reach.)"""
        A, J, _ = cell_geometry(self.net.X, self.net.topo)
        bad = (A <= 0) | (J < 0.155 * A ** 2)
        self._bad_for = self._bad_for + 1 if bad.any() else 0
        if self._bad_for > self.FOLD_PATIENCE or abs(A.sum() - L * L) > 1e-6:
            raise InvertedCellError(
                f"the network got tangled at step {self.step_no}; try another seed, "
                f"a larger gamma (now {self.p.gamma}) or fewer columns")

    def params_dict(self):
        return asdict(self.p)
