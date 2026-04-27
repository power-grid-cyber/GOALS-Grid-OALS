"""
datacenter_registry.py
======================
Central registry. Any study imports this module and calls get_datacenter()
to obtain a study-appropriate adapter without touching physics internals.

Usage
-----
    from datacenter_registry import register, get_datacenter

    register("DC_FEEDER_634", config={"seed": 42})
    dc = get_datacenter("DC_FEEDER_634", "distribution")
    P_mw, Q_mvar = dc.step(V_pcc_pu, freq_hz, t, dt)
"""

from datacenter_core import DatacenterPhysics
from adapters import (
    DistributionAdapter,
    TransmissionAdapter,
    OPFAdapter,
    MarketAdapter,
)

_REGISTRY: dict = {}

_ADAPTER_MAP = {
    'distribution': DistributionAdapter,
    'transmission': TransmissionAdapter,
    'opf':          OPFAdapter,
    'market':       MarketAdapter,
}


def register(name: str, config: dict = None) -> DatacenterPhysics:
    """
    Instantiate and register a datacenter physics model.

    Parameters
    ----------
    name   : unique identifier used by get_datacenter()
    config : optional configuration dict passed to DatacenterPhysics

    Returns the DatacenterPhysics instance (rarely needed directly).
    """
    physics = DatacenterPhysics(config or {})
    _REGISTRY[name] = physics
    print(f"  [Registry] Registered datacenter '{name}'")
    return physics


def get_datacenter(name: str, study_type: str, **kwargs):
    """
    Return a study-type adapter for a registered datacenter.

    Parameters
    ----------
    name       : name used in register()
    study_type : 'distribution' | 'transmission' | 'opf' | 'market'
    **kwargs   : forwarded to the adapter constructor

    Raises
    ------
    KeyError   if name has not been registered
    ValueError if study_type is unknown
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Datacenter '{name}' not registered. "
            f"Call register('{name}') first. "
            f"Registered: {list(_REGISTRY)}")
    if study_type not in _ADAPTER_MAP:
        raise ValueError(
            f"Unknown study type '{study_type}'. "
            f"Available: {list(_ADAPTER_MAP)}")
    return _ADAPTER_MAP[study_type](_REGISTRY[name], **kwargs)


def list_registered() -> list:
    return list(_REGISTRY.keys())


def deregister(name: str):
    """Remove a datacenter instance (e.g. between Monte Carlo scenarios)."""
    _REGISTRY.pop(name, None)
