# Basalt columns: from T junctions to hexagons

A simulation of the crack network in a cooling lava flow, in two phases:

1. **The surface cracks.** It starts from intact crust, with no cracks and no seed
   pattern. Cracks appear where the crust is most stretched, grow, and stop when
   they reach an older crack, so every junction is born as a T junction (90° / 180°).
2. **The cracks grow downward.** Slice by slice, the T junctions turn into
   Y junctions (about 120°), and hexagons become the most common column shape.

Nothing in the code says "hexagon", "6" or "120°". The cracks only follow
physical rules and two energies, and the hexagons are measured afterwards.

## Run it

```
pip install numpy scipy matplotlib pillow
python control_panel.py          # interactive window with sliders and buttons
python make_movie.py             # no window: writes GIF + figures to ./output
python make_movie.py --pattern voronoi --seed 7 --cells 500 --steps 3000

pip install pyvista               # only needed for the 3D view
python view3d.py                  # rotatable 3D window of the columns
python view3d.py --seed 5 --steps 400 --window 0.7
python view3d.py --gif basalt_3d.gif   # no window: GIF of the columns growing
```

## Files (read them in this order)

| File | What it contains |
|---|---|
| `physics.py` | The energy (elastic + crack) and its exact gradient. All the math. |
| `network.py` | The crack network: junctions, columns, one depth step, T1 swaps, pinch-outs |
| `crack_growth.py` | Phase 1: cracks growing on the intact surface, and turning them into columns |
| `patterns.py` | All starting surfaces: growing cracks (default), splitting, random Voronoi, brick wall |
| `measure.py` | Where hexagons and junction angles are *detected* |
| `simulation.py` | `Params` + `Simulation`: runs phase 1, then steps the network downward |
| `plots.py` | Drawing only: live dashboard, summary and snapshot figures, GIF writer |
| `control_panel.py` | Interactive window |
| `make_movie.py` | Command-line runner |
| `view3d.py` | Rotatable 3D view of the columns (needs PyVista) |

## Control panel

| Control | Effect |
|---|---|
| Columns, Seed, Start pattern | Starting surface. Applied when you press **Reset** (R). |
| Watch the surface crack first | With "Growing cracks": show phase 1 before going down. Untick to skip straight to the depth phase. |
| Crack energy γ | Cost of crack length relative to elastic energy. Applies immediately. |
| Step size dt | How far junctions move per depth step. Applies immediately. |
| Speed | Steps computed per screen refresh (phase 1 runs at a fifth of this). |
| Total depth | Where the run stops. You can raise it mid-run. |
| Buttons | Run/Pause (space), Step +1 (S), Reset (R), New seed (N), Save PNG (P), 3D view |

**3D view** opens `view3d.py` in its own window for the current pattern, columns,
seed, γ and dt (up to 300 depth steps; the T → Y change happens in the first ~200).

## 3D view (`view3d.py`)

The program runs the same simulation and keeps the crack outline every few
depth steps. Each corner of a column is joined to the same corner one slice
lower, so the walls are continuous; where a column changes its number of
corners it takes a small step instead. Colour = number of corners at that depth. The top of the block is the cracked
surface (T junctions, mostly 4 and 5 corners); going down the sides, the colour
turns blue as the columns become hexagons.

| Control | Effect |
|---|---|
| Left-drag / wheel / middle-drag | Rotate / zoom / pan |
| Play (button or space) | Grow the columns down from the surface, slice by slice |
| "grown down to" slider | How deep the block reaches |
| "cut from the top" slider | Remove the top part, to see a cross-section at that depth |
| r / b / s | Reset view / view from below / save a screenshot |

Options: `--pattern --cells --seed --gamma --dt` as in `make_movie.py`;
`--steps` (depth, default 300); `--window` (part of the patch shown, default 0.5);
`--height` (height of the block relative to its width, default 1; the vertical
scale is a free choice, real columns are many times taller than wide);
`--box` (cut the columns at the window edges); `--screenshot file.png` (no window);
`--gif file.gif` (grow + slow spin, no window); `--no-cache` (rerun the simulation).

Speed: the simulation result is saved in `~/.cache/basalt_sim`, so opening the
same settings again (also from the panel's 3D view button) skips the simulation.
The block is a single mesh; the sliders and Play only cut prepared arrays, and
the window uses FXAA instead of 8x multisampling.

| (software rendering, 300 columns) | before | now |
|---|---|---|
| Open the window, first time | 13.4 s | 8.2 s |
| Open again, same settings | 13.4 s | 3.4 s |
| One frame while rotating | 994 ms | 148 ms |
| One slider / Play step | 1250 ms | 115 ms |

(With a graphics card the frame times are a few ms either way; the ratio is what matters.)
Each column keeps the same id from the surface down (`CrackNetwork.cell_ids`),
which is how the slices are stacked into continuous columns.

## Phase 1: how the surface cracks (no seed)

At step 0 there are no cracks. Three rules, applied over and over:

1. **Start where the crust is most stretched.** Next to a crack the crust has relaxed,
   so a new crack starts at the point farthest from all existing cracks (best of a
   few random candidate points, because rock has random flaws).
2. **Grow across the largest pull.** Each crack grows from both tips. Near an older
   crack, the pull across that crack is relieved, so the largest pull runs along it.
   A crack runs across the largest pull, so the tip turns toward the older crack's
   normal. The closer it is, the harder it turns.
3. **Stop at an older crack.** Nothing can pull across a gap, so the tip stops there.
   Because of rule 2 it usually arrives close to 90°: a **T junction**.

In 6 runs with 300 columns, the median meeting angle was 81–82°, and 85% of all
crack tips met an older crack at more than 60°. The finished crack lines are then turned
into columns: the junctions become corners, and curved cracks keep a few bend
points so that their shape is not lost.

## Phase 2: the model in five lines

1. **Elastic energy.** A column shrinks toward its centroid while glued to hot rock below.
   E ∝ ∫|x − c|² dA = J (polar second moment of area).
2. **Crack energy.** Opening a crack costs energy per unit length (Griffith): E = Γ · length.
3. **Total.** E = Σ J/a0² + γ Σ length/l0.
4. **Growth with depth.** Each step, every junction moves a little downhill: dX/dz = −dE/dX.
   The gradient is derived by hand and checked against finite differences.
5. **Topology.** When a crack shrinks to ~0, its junctions swap neighbours (T1).
   A column that becomes tiny, paper-thin or twisted ends (pinch-out). Bend points
   left from curved surface cracks are dropped once the crack has straightened.

## Other starting surfaces

- **Splitting pieces:** a faster shortcut to a similar T-junction surface. The most
  stressed piece is cut straight across its long axis, again and again. It needs a
  few first cracks to start from (16 random regions).
- **Random points (Voronoi):** fully random, junctions at random angles, no
  T junctions. It shows that the end result does not depend on the start.
- **Brick wall:** the seed used in the first version (long straight horizontal
  cracks). Its alignment is an artefact, kept for comparison.

## Typical results (300 columns, γ = 0.1, 1500 steps, 8 seeds per start)

| Start | T junctions at start | Hexagons at start | Hexagons at end | Largest junction angle at end |
|---|---|---|---|---|
| Growing cracks (default) | 97–99% | 3–5% | 55–60% | ~128° |
| Splitting pieces | 96–98% | 0–1% | 53–61% | ~128° |
| Random points (Voronoi) | 46–53% | 24–34% | 54–64% | ~128° |
| Brick wall | 100% | 0% | 54–64% | ~128° |

"T junction" here means the largest of its three angles is above 150°.
The shape factor ⟨J/A²⟩ falls from about 0.19–0.21 to 0.163. A perfect hexagon
gives 0.1604 and a square gives 0.1667. A lower γ (cheaper cracks) gives more
hexagons. A higher γ gives more 5- and 7-sided columns, and more columns pinch out.
