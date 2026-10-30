"""Neutral beams in tokamak and stellarator plasmas: deposition, shine-through, slowing down."""

from freeofcharge.beam import NeutralBeam, tangential_injection
from freeofcharge.deposition import Deposition, deposit
from freeofcharge.equilibrium import Equilibrium, FluxSurfaces, MagneticEquilibrium
from freeofcharge.profiles import IonSpecies, Plasma
from freeofcharge.slowing_down import SlowingDown, slow_down
from freeofcharge.stopping import ConstantStopping, StoppingModel, SuzukiStopping

__version__ = "0.1.0"

__all__ = [
    "ConstantStopping",
    "Deposition",
    "Equilibrium",
    "FluxSurfaces",
    "IonSpecies",
    "MagneticEquilibrium",
    "NeutralBeam",
    "Plasma",
    "SlowingDown",
    "StoppingModel",
    "SuzukiStopping",
    "deposit",
    "slow_down",
    "tangential_injection",
]
