"""Neutral beams in tokamak and stellarator plasmas: deposition, shine-through, slowing down."""

import scope_profiler as sp

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.beam import NeutralBeam, tangential_injection
from freeofcharge.coils import CoilField, CoilSet
from freeofcharge.deposition import Deposition, deposit
from freeofcharge.equilibrium import (
    Equilibrium,
    FluxSurfaces,
    MagneticEquilibrium,
    monte_carlo_volumes,
    rho_coordinate,
)
from freeofcharge.gridded_equilibrium import GriddedEquilibrium
from freeofcharge.injectors import (
    BeamSource,
    InjectorBox,
    SourceDepositions,
    deposit_sources,
)
from freeofcharge.orbits import OrbitFollowing, follow_orbits
from freeofcharge.profiles import IonSpecies, Plasma
from freeofcharge.slowing_down import SlowingDown, slow_down
from freeofcharge.stopping import ConstantStopping, StoppingModel, SuzukiStopping

__version__ = "0.1.0"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)

__all__ = [
    "BeamSource",
    "CoilField",
    "CoilSet",
    "ConstantStopping",
    "Deposition",
    "Equilibrium",
    "FluxSurfaces",
    "GriddedEquilibrium",
    "InjectorBox",
    "IonSpecies",
    "MagneticEquilibrium",
    "NeutralBeam",
    "OrbitFollowing",
    "Plasma",
    "SlowingDown",
    "SourceDepositions",
    "StoppingModel",
    "SuzukiStopping",
    "deposit",
    "deposit_sources",
    "follow_orbits",
    "monte_carlo_volumes",
    "rho_coordinate",
    "slow_down",
    "tangential_injection",
]
