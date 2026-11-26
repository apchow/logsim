"""Deterministic cached routing adapters (Planar, OSRM, GraphHopper)."""

from app.routing.adapters import (
    PROFILE_SPECS,
    GeoPoint,
    GraphHopperClientAdapter,
    OSRMClientAdapter,
    PlanarCachedAdapter,
    RouteMatrix,
    RoutingAdapter,
    compute_matrix_sha256,
    get_route_matrix,
    resolve_profile_spec,
)

__all__ = [
    "PROFILE_SPECS",
    "GeoPoint",
    "GraphHopperClientAdapter",
    "OSRMClientAdapter",
    "PlanarCachedAdapter",
    "RouteMatrix",
    "RoutingAdapter",
    "compute_matrix_sha256",
    "get_route_matrix",
    "resolve_profile_spec",
]
