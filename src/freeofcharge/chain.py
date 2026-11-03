"""The stages of a negative-ion neutral-beam injector, from the source to the duct.

Each stage is a small function on plain numbers, so an injector design can be scanned
end to end (see ``examples/injector_chain.py``); the plasma side is freeofcharge's own
`freeofcharge.deposit` and `freeofcharge.slow_down`. The stages use the
other max-models codes, imported only when called (``pip install 'freeofcharge[chain]'``):

1. `deuterium_source`: the plasma of a deuterium RF source from a
   `wellbalanced.GlobalModel`, and the negative-ion current density it supplies,
   taken as a share ``extraction_ratio`` of the positive-ion Bohm flux to the plasma grid.
   The negative-ion surface production on caesiated walls is not modelled: that share is
   an input.
2. `slit_divergence`: the rms divergence of a slit beamlet, and the share the
   extraction electrode lets through, against its normalized perveance ``j / j_CL``,
   from tree-d-3d's boundary-integral electrodes and meshless space charge (a gun-code
   fixed point, after tree-d-3d's ``examples/pierce_gun.py``). The plasma meniscus is
   replaced by a fixed planar emitter between focusing plasma-electrode wings, so the
   V-curve comes from the electrode focusing against the space charge, not from a
   moving meniscus.
3. `neutral_fraction`: the neutral share of a negative-ion beam after a gas
   neutralizer of line density ``pi``, from `plasmacoll.charge_state_fractions`,
   checked by `neutral_fraction_closed_form`.
4. `duct_transmission`: the share of a Gaussian beam that passes a square duct.

The physical arguments of the stages take astropy Quantities (``9 * u.kV``),
strings such as ``"0.3 Pa"`` or plain numbers in SI units, eV for the gas
temperature (or K) and rad for the divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from methodcard import Rule, method
from plasmacoll.units import to_array, to_unit
from scipy.special import erf

if TYPE_CHECKING:
    from wellbalanced import GlobalState

ELEMENTARY_CHARGE = 1.602176634e-19
EPSILON_0 = 8.8541878128e-12
ATOMIC_MASS = 1.66053906660e-27
BOLTZMANN = 1.380649e-23
DEUTERIUM_MASS = 2.014 * ATOMIC_MASS

#: Charge-changing cross sections (m^2) of a 1 MeV D- beam on D2 gas: single stripping,
#: double stripping and re-ionization of the neutrals. Round numbers of plausible size
#: (wellbalanced's illustrative neutralizer), not data: they give a neutral maximum of
#: about 58 % near 1.4e20 m^-2, close to the 55-60 % quoted for gas neutralizers of
#: 1 MeV D- beams (Hemsworth et al., Nucl. Fusion 49, 045006, 2009). Replace with
#: evaluated data (ORNL Redbook, IAEA ALADDIN) for design work.
GAS_NEUTRALIZER = {("-", "0"): 1.2e-20, ("-", "+"): 5.0e-22, ("0", "+"): 3.7e-21}


def _child_langmuir_rules() -> dict:
    """The SymPy rule of :func:`child_langmuir` (for the tests)."""
    import sympy

    v, d, m = sympy.symbols("voltage gap mass", positive=True)
    eps0, e = sympy.Float(EPSILON_0), sympy.Float(ELEMENTARY_CHARGE)
    rule = (
        sympy.Rational(4, 9)
        * eps0
        * sympy.sqrt(2 * e / m)
        * v ** sympy.Rational(3, 2)
        / d**2
    )
    ranges = {
        "voltage": (1.0e3, 1.0e4),
        "gap": (1.0e-3, 1.0e-2),
        "mass": (1.6e-27, 5.1e-27),
    }
    return {"current": Rule(rule, ranges=ranges, rtol=1e-9)}


@method(
    equations={
        "current": r"j_\mathrm{CL} = \frac{4\varepsilon_0}{9}\sqrt{\frac{2e}{m}}\, \frac{V^{3/2}}{d^2}",
    },
    assumptions=(r"planar diode, singly charged ions; the deuteron mass by default",),
    symbolic=lambda: _child_langmuir_rules(),
)
def child_langmuir(voltage: float, gap: float, mass: float = DEUTERIUM_MASS) -> float:
    """The planar space-charge-limited current density (A/m^2) of singly charged ions.

    ``voltage`` [V], ``gap`` [m] and the ion ``mass`` [kg] take quantities
    (``"9 kV"``, ``"6 mm"``).
    """
    voltage = to_array(voltage, "V", "voltage")
    gap = to_array(gap, "m", "gap")
    mass = to_array(mass, "kg", "mass")
    return (
        4.0
        / 9.0
        * EPSILON_0
        * np.sqrt(2.0 * ELEMENTARY_CHARGE / mass)
        * voltage**1.5
        / gap**2
    )


@dataclass(frozen=True)
class Source:
    """The plasma of the source and the negative-ion current density it supplies.

    Attributes:
        rf_power: RF power delivered to the source [W].
        electron_density: [m^-3].
        electron_temperature: [eV].
        positive_flux: Positive-ion flux density to the plasma grid, ``h n_e u_B``
            [m^-2 s^-1].
        current_density: Extracted negative-ion current density [A/m^2].
        state: The steady state of the global model, with every term of its balances.
    """

    rf_power: float
    electron_density: float
    electron_temperature: float
    positive_flux: float
    current_density: float
    state: GlobalState


@method(
    equations={
        "rates": r"k = a\,e^{-c/T_e}",
        "current": r"j_- = \text{extraction\_ratio}\cdot e\, h\, n_{D_2^+} u_B, \qquad u_B = \sqrt{\frac{e T_e}{m_{D_2^+}}}",
    },
    assumptions=(
        r"a wellbalanced global model of e, D$_2$, D, D$_2^+$ at fixed D$_2$ density $p/(e T_g)$",
        r"illustrative Arrhenius rates, not data: ionisation ($a = 2\cdot10^{-14}$ m³/s, $c = 15.4$ eV), dissociation ($10^{-14}$, 8.8 eV), elastic ($3\cdot10^{-14}$ m³/s)",
        r"the negative-ion surface production is an input share (`extraction_ratio`)",
    ),
)
def deuterium_source(
    rf_power: float,
    *,
    pressure: float = 0.3,
    volume: float = 0.4,
    wall_area: float = 6.0,
    h_factor: float = 0.3,
    absorbed_fraction: float = 0.5,
    extraction_ratio: float = 0.3,
    gas_temperature: float = 0.05,
    recombination: float = 0.1,
) -> Source:
    """The steady plasma of a deuterium RF source, and its negative-ion current density.

    A `wellbalanced.GlobalModel` of D2 at ``pressure`` [Pa] in a chamber of
    ``volume`` [m^3] and ``wall_area`` [m^2] absorbs ``absorbed_fraction`` of
    ``rf_power``. Its reactions (ionization and dissociation of D2, elastic collisions)
    use Arrhenius rates of plausible size, wellbalanced's illustrative deuterium set,
    not data; the atoms recombine to D2 at the walls with probability
    ``recombination``. The negative-ion current density is ``extraction_ratio`` times the
    charge flux of the positive ions to the grid, ``e h n_e u_B`` (``u_B`` of D2+).

    ``rf_power`` [W], ``pressure`` [Pa], ``volume`` [m^3], ``wall_area`` [m^2] and
    ``gas_temperature`` [eV, or K] take quantities (``"200 kW"``, ``"300 K"``); the
    other arguments are shares.
    """
    rf_power = to_unit(rf_power, "W", "rf_power")
    pressure = to_unit(pressure, "Pa", "pressure")
    volume = to_unit(volume, "m^3", "volume")
    wall_area = to_unit(wall_area, "m^2", "wall_area")
    gas_temperature = to_unit(gas_temperature, "eV", "gas_temperature")
    from plasmacoll.constants import ELECTRON_MASS
    from wellbalanced import (
        ELECTRON,
        Chamber,
        GlobalModel,
        Reaction,
        Species,
        WallReaction,
    )

    d2 = Species("D2", 2 * DEUTERIUM_MASS)
    d = Species("D", DEUTERIUM_MASS)
    d2_ion = Species("D2+", 2 * DEUTERIUM_MASS - ELECTRON_MASS, 1)

    def arrhenius(a: float, c: float):
        return lambda t: a * np.exp(-c / np.asarray(t, dtype=float))

    gas_density = pressure / (ELEMENTARY_CHARGE * gas_temperature)  # p = n k T_g
    model = GlobalModel(
        species=[ELECTRON, d2, d, d2_ion],
        reactions=[
            Reaction(
                ("e", "D2"),
                ("e", "e", "D2+"),
                arrhenius(2.0e-14, 15.4),
                electron_energy_loss=15.4,
                name="ionization of D2",
            ),
            Reaction(
                ("e", "D2"),
                ("e", "D", "D"),
                arrhenius(1.0e-14, 8.8),
                electron_energy_loss=8.8,
                name="dissociation of D2",
            ),
            Reaction(
                ("e", "D2"),
                ("e", "D2"),
                3.0e-14,
                elastic_mass=d2.mass,
                name="elastic e-D2",
            ),
        ],  # fmt: skip
        chamber=Chamber(volume, wall_area, h_factor=h_factor),
        power=absorbed_fraction * rf_power,
        fixed={"D2": gas_density},
        ion_wall_products={"D2+": {"D2": 1}},
        wall_reactions=[WallReaction("D", recombination, {"D2": 0.5})],
        gas_temperature=gas_temperature,
    )
    state = model.steady_state()
    if not state.converged:
        raise RuntimeError(f"the source model did not converge at {rf_power:g} W")
    te = state.electron_temperature
    u_bohm = np.sqrt(ELEMENTARY_CHARGE * te / d2_ion.mass)
    flux = h_factor * state.densities["D2+"] * u_bohm
    return Source(
        rf_power=rf_power,
        electron_density=state.electron_density,
        electron_temperature=te,
        positive_flux=flux,
        current_density=extraction_ratio * ELEMENTARY_CHARGE * flux,
        state=state,
    )


class _SlitGun:
    """The rays of one slit beamlet at normalized perveance ``p``; see ``slit_divergence``."""

    gap, voltage, half_width, start = 1.0, 1.0, 0.5, 0.02
    k_e = 1.0 / (4.0 * np.pi)  # eps0 = 1 for the log kernel

    def __init__(
        self,
        p: float,
        *,
        rays: int,
        panel: float,
        dt: float,
        angle: float = 50.0,
        aperture: float = 2.0,
    ) -> None:
        from simgeometry import Aperture2D, Polyline
        from tree_d_3d import BoundaryIntegralElectrodes, Quad, conductors_from_shape

        w, d = self.half_width, self.gap
        wing = 0.8 * d
        angle = np.radians(angle)
        dx, dy = wing * np.cos(angle), wing * np.sin(angle)
        plasma_electrode = Polyline(
            np.array([(dx, -w - dy), (0.0, -w), (0.0, w), (dx, w + dy)])
        )
        self.aperture = aperture * w
        extraction = Aperture2D.slit(
            d + 0.1, half_width=self.aperture, thickness=0.2, length=2.0
        )
        conductors = conductors_from_shape(
            plasma_electrode, 0.0, panel
        ) + conductors_from_shape(extraction, self.voltage, panel)
        self.bem = BoundaryIntegralElectrodes(conductors, k_e=self.k_e)
        self.end = d + 0.2 + d  # one gap past the extraction electrode
        j_cl = 4.0 / 9.0 * np.sqrt(2.0) * self.voltage**1.5 / d**2
        self.y0 = w * (2 * (np.arange(rays) + 0.5) / rays - 1)
        self.d_i = p * j_cl * 2 * w / rays
        self.softening = w / rays
        self.root = Quad(0.5 * self.end, 0.0, self.end)
        self.dt = dt
        self.steps = int(np.ceil(3 * self.end / (np.sqrt(2 * self.voltage) * dt)))
        self.rays = rays
        self.x = None
        self.v0 = None

    def _sources(self):
        if self.x is None:
            return np.zeros((2, 0)), np.zeros(0)
        a, b = self.x[:-1], self.x[1:]
        ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
        mid = 0.5 * (a + b)
        xy = np.stack((mid[:, 0][ok], mid[:, 1][ok]))
        q = np.full(xy.shape[1], -self.d_i * self.dt)
        layer = np.stack((np.full(self.rays, self.start / 4), self.y0))
        q_layer = -self.d_i * 3 * self.start / self.v0
        return np.concatenate((xy, layer), axis=1), np.concatenate((q, q_layer))

    def _field(self, points, tree):
        from tree_d_3d.flat_tree import direct_field

        _, e = direct_field(points, self.bem.midpoints, self.bem.charge, kernel="log")
        e = self.k_e * e
        if tree is not None:
            _, e_b = tree.evaluate(points, theta=0.5, softening=self.softening)
            e = e + self.k_e * np.asarray(e_b)
        return e

    def _phi(self, points, tree):
        from tree_d_3d.flat_tree import direct_field

        phi, _ = direct_field(points, self.bem.midpoints, self.bem.charge, kernel="log")
        phi = self.k_e * phi
        if tree is not None:
            phi_b, _ = tree.evaluate(points, theta=0.5, softening=self.softening)
            phi = phi + self.k_e * np.asarray(phi_b)
        return phi

    def divergence(self, iterations: int, omega: float = 0.5) -> tuple[float, float]:
        from tree_d_3d import FlatQuadtree

        final = np.full((2, self.rays), np.nan)
        for _ in range(iterations):
            xy, q = self._sources()
            self.bem.solve(xy, q)
            tree = (
                FlatQuadtree(xy, q, self.root, kernel="log", order=12, leaf_capacity=32)
                if q.size
                else None
            )
            start = np.stack((np.full(self.rays, self.start), self.y0))
            v0 = np.sqrt(2 * np.maximum(self._phi(start, tree), 1e-4 * self.voltage))
            x = np.full((self.steps + 1, 2, self.rays), np.nan)
            pos, vel = start.copy(), np.stack((v0, np.zeros(self.rays)))
            x[0] = pos
            final = np.full((2, self.rays), np.nan)
            alive = np.ones(self.rays, dtype=bool)
            e = self._field(pos, tree)
            for i in range(self.steps):
                vel = vel - 0.5 * self.dt * e  # q / m = -1: negative ions toward +V
                pos = pos + self.dt * vel
                e = self._field(pos, tree)
                vel = vel - 0.5 * self.dt * e
                done = alive & (pos[0] > self.end)
                final[:, done] = vel[:, done]
                alive &= (pos[0] >= 0.0) & (pos[0] <= self.end)
                alive &= np.abs(pos[1]) < self.aperture  # else it hits a plate
                x[i + 1][:, alive] = pos[:, alive]
                if not alive.any():
                    break
            if self.x is not None:
                x = np.where(
                    np.isfinite(self.x) & np.isfinite(x),
                    (1 - omega) * self.x + omega * x,
                    x,
                )
            self.x, self.v0 = x, v0
        angles = final[1] / final[0]
        through = np.isfinite(angles)  # the others were intercepted
        if not through.any():
            return float("nan"), 0.0
        return float(np.sqrt(np.mean(angles[through] ** 2))), float(through.mean())


@method(
    equations={
        "divergence": r"\theta_\mathrm{rms} = \sqrt{\left\langle (v_y / v_x)^2 \right\rangle} \quad \text{one gap past the extraction electrode}",
        "injection": r"v_0 = \sqrt{2\phi}, \qquad j = \text{perveance}\cdot j_\mathrm{CL}",
    },
    assumptions=(
        r"code units $\varepsilon_0 = |q|/m = d = V = 1$; tree-d-3d's boundary-integral electrodes and meshless space charge",
        r"a fixed point of rays, line charges and electrodes, under-relaxed with $\omega = 0.5$; kick-drift-kick steps",
        r"converges up to a perveance of about 0.5 at the default resolution",
    ),
    components=(child_langmuir,),
)
def slit_divergence(
    perveances,
    *,
    rays: int = 24,
    iterations: int = 8,
    panel: float = 0.05,
    dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """The rms divergence [rad] of a slit beamlet, and its transmitted share, per perveance.

    Code units ``eps0 = |q| / m = 1``, gap ``d = 1`` and extraction voltage ``V = 1``;
    angles do not depend on the scale. The ions start at the emitter, a strip of
    half-width ``w = 0.5`` between plasma-electrode wings at 50 degrees from the axis
    (steeper than Pierce's 67.5, so that the vacuum field focuses; potential 0), with
    the current density ``perveance * J_CL`` across it; they are accelerated over the
    gap to an extraction electrode at ``V`` (a slit of half-width ``2 w``, thickness
    ``0.2``). The steady beam is the fixed point of tracing the rays, turning them into
    line charges for tree-d-3d's tree code and solving the electrodes' response,
    under-relaxed. The divergence is the rms of ``v_y / v_x`` of the rays one gap past
    the extraction electrode; rays that reach a plate first are intercepted, and the
    transmitted share is the fraction of rays (of the current) that get through. With
    the default resolution the fixed point converges up to a perveance of about 0.5;
    beyond, the beam reaches the extraction electrode and the iteration no longer
    settles, so do not trust the results there.
    """
    perveances = np.atleast_1d(np.asarray(perveances, dtype=float))
    results = [
        _SlitGun(p, rays=rays, panel=panel, dt=dt).divergence(iterations)
        for p in perveances
    ]
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])


@method(
    equations={
        "rate_equations": r"\frac{dF_i}{d\pi} = \sum_{j \ne i} \left(\sigma_{ji} F_j - \sigma_{ij} F_i\right), \qquad F_-(0) = 1",
    },
    assumptions=(
        r"thin target: the fractions depend on the line density $\pi$ only",
        r"all of the beam negative at the entrance; placeholder cross sections (`GAS_NEUTRALIZER`)",
    ),
    references=(
        "[Hemsworth et al., Nucl. Fusion 49, 045006 (2009)](https://iopscience.iop.org/0029-5515/49/4/045006)",
    ),
)
def neutral_fraction(line_density, cross_sections=None) -> np.ndarray:
    """The neutral share of a D- beam after a gas target of ``line_density`` [m^-2].

    From `plasmacoll.charge_state_fractions` with ``cross_sections`` [m^2]
    (`GAS_NEUTRALIZER` by default), all of the beam negative at the entrance.
    Both take quantities (``"1.4e20 m^-2"``).
    """
    line_density = to_array(line_density, "m^-2", "line_density")
    if cross_sections is not None:
        cross_sections = to_unit(cross_sections, "m^2", "cross_sections")
    from plasmacoll import ChargeStateTransitions, charge_state_fractions

    pis = np.atleast_1d(np.asarray(line_density, dtype=float))
    order = np.argsort(pis)
    path = np.concatenate(([0.0], pis[order]))
    transitions = ChargeStateTransitions(
        {"gas": dict(GAS_NEUTRALIZER if cross_sections is None else cross_sections)}
    )
    fractions = charge_state_fractions(transitions, {"gas": path})[1:, 1]
    result = np.empty_like(pis)
    result[order] = fractions
    return result


def _closed_form_rules() -> dict:
    """The SymPy rule of `neutral_fraction_closed_form` (for the tests)."""
    import sympy

    s1, s2, s3, pi = sympy.symbols("s1 s2 s3 line_density", positive=True)
    rule = s1 / (s1 + s2 - s3) * (sympy.exp(-s3 * pi) - sympy.exp(-(s1 + s2) * pi))
    return {
        "neutral": Rule(
            rule,
            implementation=lambda s1, s2, s3, line_density: (
                neutral_fraction_closed_form(
                    line_density, {("-", "0"): s1, ("-", "+"): s2, ("0", "+"): s3}
                )
            ),
            ranges={
                "s1": (1.0e-20, 2.0e-20),
                "s2": (1.0e-22, 1.0e-21),
                "s3": (2.0e-21, 5.0e-21),
                "line_density": (1.0e19, 1.0e21),
            },
        )
    }


@method(
    equations={
        "negative": r"F_- = e^{-(s_1 + s_2) \pi}",
        "neutral": r"F_0 = \frac{s_1}{s_1 + s_2 - s_3} \left(e^{-s_3 \pi} - e^{-(s_1 + s_2) \pi}\right)",
    },
    assumptions=(
        r"no electron capture ($+ \to 0$, $0 \to -$)",
        r"$s_1$ single and $s_2$ double stripping, $s_3$ re-ionisation; $\pi$ the line density",
    ),
    symbolic=_closed_form_rules,
)
def neutral_fraction_closed_form(line_density, cross_sections=None) -> np.ndarray:
    """The neutral share in closed form, with no electron capture (``+ -> 0``, ``0 -> -``).

    ``F_- = exp(-(s1 + s2) pi)`` and
    ``F_0 = s1 / (s1 + s2 - s3) (exp(-s3 pi) - exp(-(s1 + s2) pi))``, with ``s1`` the
    single stripping, ``s2`` the double stripping and ``s3`` the re-ionization.
    ``line_density`` [m^-2] and ``cross_sections`` [m^2] take quantities.
    """
    line_density = to_array(line_density, "m^-2", "line_density")
    if cross_sections is not None:
        cross_sections = to_unit(cross_sections, "m^2", "cross_sections")
    sigma = GAS_NEUTRALIZER if cross_sections is None else cross_sections
    s1, s2, s3 = sigma[("-", "0")], sigma.get(("-", "+"), 0.0), sigma[("0", "+")]
    pi = np.asarray(line_density, dtype=float)
    return s1 / (s1 + s2 - s3) * (np.exp(-s3 * pi) - np.exp(-(s1 + s2) * pi))


def _duct_rules() -> dict:
    """The SymPy rule of :func:`duct_transmission` (for the tests)."""
    import sympy

    w, theta, length, a = sympy.symbols(
        "width divergence length half_aperture", positive=True
    )
    sigma = sympy.sqrt(w**2 + (length * theta) ** 2)
    rule = sympy.erf(a / (sympy.sqrt(2) * sigma)) ** 2
    ranges = {
        "width": (0.01, 0.1),
        "divergence": (1.0e-3, 1.0e-2),
        "length": (1.0, 20.0),
        "half_aperture": (0.05, 0.3),
    }
    return {"transmission": Rule(rule, ranges=ranges)}


@method(
    equations={
        "width": r"\sigma(L) = \sqrt{w^2 + (L\,\theta)^2}",
        "transmission": r"T = \mathrm{erf}\!\left(\frac{a}{\sqrt 2\, \sigma(L)}\right)^2",
    },
    assumptions=(
        r"a round Gaussian beam through a square aperture: the product of two independent one-dimensional shares",
    ),
    symbolic=lambda: _duct_rules(),
)
def duct_transmission(
    width: float, divergence: float, length: float, half_aperture: float
) -> float:
    """The share of a round Gaussian beam that passes a square duct.

    The beam has standard deviation ``width`` [m] at the source and ``divergence``
    [rad]; at ``length`` [m] its width is ``sqrt(width^2 + (length divergence)^2)``, and
    a square aperture of ``half_aperture`` [m] keeps ``erf(a / (sqrt(2) sigma))^2``.
    Each takes a quantity (``"10 cm"``, ``"5 mrad"``).
    """
    width = to_array(width, "m", "width")
    divergence = to_array(divergence, "rad", "divergence")
    length = to_array(length, "m", "length")
    half_aperture = to_array(half_aperture, "m", "half_aperture")
    sigma = np.hypot(width, length * divergence)
    return float(erf(half_aperture / (np.sqrt(2.0) * sigma)) ** 2)
