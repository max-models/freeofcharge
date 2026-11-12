"""Neutral beams in tokamak and stellarator plasmas: deposition, shine-through, slowing down."""

from freeofcharge.beam import NeutralBeam, tangential_injection
from freeofcharge.coils import CoilField, CoilSet
from freeofcharge.deposition import Deposition, deposit
from freeofcharge.equilibrium import (
    Equilibrium,
    FluxSurfaces,
    MagneticEquilibrium,
    rho_coordinate,
)
from freeofcharge.gridded_equilibrium import GriddedEquilibrium
from freeofcharge.injectors import (
    BeamSource,
    InjectorBox,
    SourceDepositions,
    deposit_sources,
)
from freeofcharge.profiles import IonSpecies, Plasma
from freeofcharge.slowing_down import SlowingDown, slow_down
from freeofcharge.stopping import ConstantStopping, StoppingModel, SuzukiStopping

__version__ = "0.1.0"

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
    "Plasma",
    "SlowingDown",
    "SourceDepositions",
    "StoppingModel",
    "SuzukiStopping",
    "deposit",
    "deposit_sources",
    "rho_coordinate",
    "slow_down",
    "tangential_injection",
]
