"""Neutral beams in tokamak and stellarator plasmas: deposition, shine-through, slowing down."""

import scope_profiler as sp

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.beam import NeutralBeam, tangential_injection
from freeofcharge.charge_exchange import (
    ChargeExchangeSlowingDown,
    slow_down_with_charge_exchange,
)
from freeofcharge.coils import CoilField, CoilSet
from freeofcharge.deposition import Deposition, deposit
from freeofcharge.distribution import distribution_moments, slowing_down_distribution
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
from freeofcharge.neutrons import NeutronRates, neutron_rates
from freeofcharge.orbit_width import FirstOrbitDeposition, orbit_average_births
from freeofcharge.orbits import OrbitFollowing, follow_orbits
from freeofcharge.profiles import IonSpecies, Plasma
from freeofcharge.run import run_nbi
from freeofcharge.shine_through import shine_through_loads
from freeofcharge.slowing_down import SlowingDown, slow_down
from freeofcharge.stopping import ConstantStopping, StoppingModel, SuzukiStopping
from freeofcharge.stopping_tables import TabulatedStopping
from freeofcharge.time_dependent import (
    SlowingDownResponse,
    slowing_down_response,
    time_dependent_slow_down,
)
from freeofcharge.torque import beam_torque

__version__ = "0.1.0"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)

__all__ = [
    "BeamSource",
    "ChargeExchangeSlowingDown",
    "CoilField",
    "CoilSet",
    "ConstantStopping",
    "Deposition",
    "Equilibrium",
    "FirstOrbitDeposition",
    "FluxSurfaces",
    "GriddedEquilibrium",
    "InjectorBox",
    "IonSpecies",
    "MagneticEquilibrium",
    "NeutralBeam",
    "NeutronRates",
    "OrbitFollowing",
    "Plasma",
    "SlowingDown",
    "SlowingDownResponse",
    "SourceDepositions",
    "StoppingModel",
    "SuzukiStopping",
    "TabulatedStopping",
    "beam_torque",
    "deposit",
    "deposit_sources",
    "distribution_moments",
    "follow_orbits",
    "monte_carlo_volumes",
    "neutron_rates",
    "orbit_average_births",
    "rho_coordinate",
    "run_nbi",
    "shine_through_loads",
    "slow_down",
    "slow_down_with_charge_exchange",
    "slowing_down_distribution",
    "slowing_down_response",
    "tangential_injection",
    "time_dependent_slow_down",
]
