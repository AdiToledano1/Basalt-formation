"""
control_panel.py - interactive window with sliders and buttons.
================================================================

    python control_panel.py

Controls
    Columns, Seed, Start pattern  -> describe the starting surface;
                                     press Reset (or R) to apply them
    Watch the surface crack first -> with the default "Growing cracks" start,
                                     first show the cracks forming on the
                                     intact crust (T junctions), then go down
    Crack energy gamma, Step dt   -> physics; apply immediately, even mid-run
    Speed                         -> depth steps computed per screen refresh
    Total depth                   -> where the run stops (can be raised mid-run)

    Buttons:  Run/Pause (space)   Step +1 (S)   Reset (R)
              New seed (N)        Save PNG (P)  3D view (opens a rotatable
              3D window for the current settings; needs  pip install pyvista)

Needs an interactive matplotlib backend (the default on Windows/macOS;
on Linux install python3-tk if the window does not open).
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from patterns import PATTERNS
from plots import Dashboard, plot_summary, INK2
from simulation import Simulation, Params, InvertedCellError

ACCENT = "#2a78d6"


class ControlPanel:
    def __init__(self, params: Params = Params()):
        for key, keep in (("keymap.save", ["ctrl+s"]), ("keymap.pan", []),
                          ("keymap.home", ["h", "home"])):
            plt.rcParams[key] = keep                     # free S, P, R for our shortcuts
        self.fig = plt.figure(figsize=(15, 9.6))
        try:
            self.fig.canvas.manager.set_window_title("Basalt columns - control panel")
        except AttributeError:
            pass
        self.running = False
        self.sim = Simulation(params)                    # starts by cracking the surface
        self.dash = Dashboard(self.fig, self.sim, rect=(0.02, 0.29, 0.96, 0.69))
        self._build_controls(params)
        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self._tick)
        self.timer.start()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ------------------------------------------------------------------ layout
    def _slider(self, x, y, label, lo, hi, init, step=None, fmt=None, w=0.12):
        ax = self.fig.add_axes([x, y, w, 0.024])
        s = Slider(ax, label, lo, hi, valinit=init, valstep=step, color=ACCENT,
                   valfmt=fmt)
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)
        return s

    def _build_controls(self, p):
        f = self.fig
        f.add_artist(plt.Line2D([0.02, 0.98], [0.235, 0.235], color="#dddddd", lw=1))
        f.text(0.02, 0.205, "Starting surface", fontsize=10, weight="bold", color="#222")
        f.text(0.27, 0.205, "Physics", fontsize=10, weight="bold", color="#222")
        f.text(0.52, 0.205, "Playback", fontsize=10, weight="bold", color="#222")
        f.text(0.73, 0.205, "Start pattern", fontsize=10, weight="bold", color="#222")

        self.s_cells = self._slider(0.075, 0.16, "Columns", 50, 1000, p.n_cells, 10, "%d")
        self.s_seed = self._slider(0.075, 0.115, "Seed", 0, 999, p.seed, 1, "%d")
        f.text(0.02, 0.075, "Columns, seed and pattern\napply when you press Reset",
               fontsize=8.5, color=INK2)
        ax_c = f.add_axes([0.015, 0.012, 0.22, 0.045], frameon=False)
        self.watch = CheckButtons(ax_c, ["Watch the surface crack first"], [True])
        for t in self.watch.labels:
            t.set_fontsize(8.5)

        self.s_gamma = self._slider(0.345, 0.16, "Crack energy γ", 0.05, 0.5, p.gamma, 0.01, "%.2f")
        self.s_dt = self._slider(0.345, 0.115, "Step size dt", 0.01, 0.2, p.dt, 0.01, "%.2f")
        f.text(0.27, 0.07, "γ: cost of crack length vs elastic energy\n"
               "dt: how far junctions move per depth step", fontsize=8.5, color=INK2)
        self.s_gamma.on_changed(lambda v: setattr(self.sim.p, "gamma", float(v)))
        self.s_dt.on_changed(lambda v: setattr(self.sim.p, "dt", float(v)))

        self.s_speed = self._slider(0.585, 0.16, "Speed", 1, 50, 5, 1, "%d /frame", w=0.09)
        self.s_steps = self._slider(0.585, 0.115, "Total depth", 100, 5000, p.steps, 100,
                                    "%d", w=0.09)
        self.s_steps.on_changed(self._on_steps)

        ax_r = f.add_axes([0.725, 0.02, 0.155, 0.175], frameon=False)
        keys = list(PATTERNS)
        self.radio = RadioButtons(ax_r, [PATTERNS[k] for k in keys],
                                  active=keys.index(p.pattern), activecolor=ACCENT)
        for t in self.radio.labels:
            t.set_fontsize(8.5)
        self.pattern_keys = keys

        def button(y, text, cb):
            b = Button(f.add_axes([0.885, y, 0.095, 0.029]), text, color="#f1f0ec",
                       hovercolor="#dfe9f7")
            b.label.set_fontsize(9.5)
            b.on_clicked(cb)
            return b
        self.b_run = button(0.193, "Run", self._toggle_run)
        self.b_step = button(0.160, "Step +1", self._step_once)
        self.b_reset = button(0.127, "Reset", self._reset)
        self.b_new = button(0.094, "New seed", self._new_seed)
        self.b_save = button(0.061, "Save PNG", self._save)
        self.b_3d = button(0.028, "3D view", self._open_3d)
        self.msg = f.text(0.98, 0.245, "", fontsize=9, color="#b3261e", ha="right")

    # ------------------------------------------------------------------ actions
    def _params_from_controls(self):
        return Params(pattern=self.pattern_keys[[l.get_text() for l in self.radio.labels]
                                                .index(self.radio.value_selected)],
                      n_cells=int(self.s_cells.val), seed=int(self.s_seed.val),
                      gamma=float(self.s_gamma.val), dt=float(self.s_dt.val),
                      steps=int(self.s_steps.val))

    def _toggle_run(self, _=None):
        if self.sim.done and not self.running:
            self._set_msg("Reached total depth - raise 'Total depth' or press Reset.")
            return
        self.running = not self.running
        self.b_run.label.set_text("Pause" if self.running else "Run")
        self.fig.canvas.draw_idle()

    def _advance(self, n):
        try:
            self.sim.advance(n)
        except InvertedCellError as e:
            self.running = False
            self.b_run.label.set_text("Run")
            self._set_msg(str(e))
        self.dash.update()
        self.fig.canvas.draw_idle()

    def _tick(self):
        if not self.running:
            return
        speed = int(self.s_speed.val)
        self._advance(max(1, speed // 5) if self.sim.forming else speed)
        if self.sim.done:
            self.running = False
            self.b_run.label.set_text("Run")
            self._set_msg("Reached total depth.")

    def _step_once(self, _=None):
        self.running = False
        self.b_run.label.set_text("Run")
        self._advance(1)

    def _reset(self, _=None):
        self.running = False
        self.b_run.label.set_text("Run")
        self._set_msg("")
        self.sim = Simulation(self._params_from_controls(),
                              watch_forming=self.watch.get_status()[0])
        self.dash.reset(self.sim)
        self.fig.canvas.draw_idle()

    def _new_seed(self, _=None):
        self.s_seed.set_val((int(self.s_seed.val) + 1) % 1000)
        self._reset()

    def _on_steps(self, v):
        self.sim.p.steps = int(v)
        self.dash.update()
        self.fig.canvas.draw_idle()

    def _save(self, _=None):
        p = self.sim.p
        if self.sim.forming:
            tag = f"{p.pattern}_seed{p.seed}_surface{self.sim.growth.step_no}"
            self.fig.savefig(f"basalt_panel_{tag}.png", dpi=110)
            self._set_msg(f"Saved basalt_panel_{tag}.png")
            return
        tag = f"{p.pattern}_seed{p.seed}_step{self.sim.step_no}"
        self.fig.savefig(f"basalt_panel_{tag}.png", dpi=110)
        plot_summary(f"basalt_summary_{tag}.png", self.sim.history)
        self._set_msg(f"Saved basalt_panel_{tag}.png and basalt_summary_{tag}.png")

    def _open_3d(self, _=None):
        """Open the rotatable 3D view (view3d.py) for the current settings in its
        own window.  It re-runs the same simulation (same seed), recording it."""
        import importlib.util
        import os
        import subprocess
        import sys
        if importlib.util.find_spec("pyvista") is None:
            self._set_msg("The 3D view needs PyVista: run  pip install pyvista")
            return
        p = self._params_from_controls()
        steps = min(p.steps, 300)   # the T -> Y change happens in the first ~200 steps
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "view3d.py")
        subprocess.Popen([sys.executable, script, "--pattern", p.pattern,
                          "--cells", str(p.n_cells), "--seed", str(p.seed),
                          "--gamma", str(p.gamma), "--dt", str(p.dt), "--steps", str(steps)])
        self._set_msg(f"Opening the 3D view ({steps} depth steps); it takes a few seconds")

    def _set_msg(self, text):
        self.msg.set_text(text)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        actions = {" ": self._toggle_run, "s": self._step_once, "r": self._reset,
                   "n": self._new_seed, "p": self._save}
        if event.key in actions:
            actions[event.key]()


if __name__ == "__main__":
    panel = ControlPanel()
    plt.show()
