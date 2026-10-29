"""Draw the animated figure of the docs home page, ``docs/src/assets/hero.svg``.

A top view of an ITER-sized tokamak (a Solov'ev equilibrium, R0 = 6.2 m) with a
1 MeV deuterium beam injected tangentially. freeofcharge runs the deposition:
the faint lines are beam markers, the dots fly along them to where the fast
ions are born (flashing, then fading), and a few cross the plasma and shine
through onto the inboard wall. The rings are flux surfaces at the midplane.
Rerun with ``make docs-hero`` or ``python docs/hero.py`` (a few seconds).
"""

from pathlib import Path

import numpy as np
from plasma_plots.hero import HeroFigure

import freeofcharge as foc

equilibrium = foc.Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, wall_psi_n=1.3)
origin, direction = foc.tangential_injection(5.3, 12.0, toroidal_angle=1.25)
beam = foc.NeutralBeam.gaussian(
    origin, direction, 1.0e6, 16.5e6, width=0.4, divergence=0.01, markers=400, seed=1
)
plasma = foc.Plasma.parabolic(2.0e19, 2.0e4, ions={"D": (2, 1, 1.0)})
deposition = foc.deposit(
    beam, equilibrium, plasma, step=0.05, births_per_marker=1, seed=2
)
print(f"shine-through: {100 * deposition.shine_through_fraction:.1f} %")

fig = HeroFigure(xlim=(-28.0, 12.0), ylim=(-7.5, 7.5))
phi = np.linspace(0.0, 2 * np.pi, 121)
wall = equilibrium.wall.vertices[:, 0]
for radius in (wall.min(), wall.max()):
    fig.line(radius * np.cos(phi), radius * np.sin(phi), tone="muted", width=2.5)
boundary = equilibrium.boundary.vertices
mid = boundary[np.abs(boundary[:, 1]) < 0.3, 0]
r_in, r_out = mid.min(), mid.max()
for rho in (0.25, 0.5, 0.75, 1.0):
    for radius in (6.2 - rho * (6.2 - r_in), 6.2 + rho * (r_out - 6.2)):
        fig.line(
            radius * np.cos(phi),
            radius * np.sin(phi),
            width=1.0,
            opacity=0.25 + 0.3 * rho,
        )

rng = np.random.default_rng(3)
start = np.asarray(beam.markers.positions)[:, :2]
births = np.asarray(deposition.births.positions)[:, :2]
picked = rng.choice(len(births), 28, replace=False)
for k, i in enumerate(picked):
    x0, y0 = start[i]
    x1, y1 = births[i]
    path = fig.line([x0, x1], [y0, y1], width=0.8, opacity=0.3)
    duration = 6.0
    begin = duration * k / len(picked)
    fig.tracer(path, duration=duration, offset=begin / duration, r=3.0)
    flash = fig.circle(x1, y1, 4.0, tone="high", opacity=0.0)
    flash.animate("opacity", [0, 0, 0.9, 0.2, 0], duration, begin=begin)
if deposition.wall_hits is not None and len(deposition.wall_hits.positions):
    hits = np.asarray(deposition.wall_hits.positions)[:, :2]
    for k, i in enumerate(rng.choice(len(hits), min(4, len(hits)), replace=False)):
        x0, y0 = start.mean(axis=0)
        path = fig.line(
            [x0, hits[i, 0]],
            [y0, hits[i, 1]],
            tone="muted",
            width=1.0,
            opacity=0.5,
            dash="4 6",
        )
        fig.tracer(path, duration=5.0, offset=k / 4, r=3.0, tone="muted")

fig.save(Path(__file__).parent / "src" / "assets" / "hero.svg")
