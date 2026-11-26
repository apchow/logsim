"""Deterministic cached routing adapters (Planar, OSRM, GraphHopper)."""

from app.routing.adapters import (
    GeoPoint,
    GraphHopperClientAdapter,
    OSRMClientAdapter,
    PlanarCachedAdapter,
    RouteMatrix,
    RoutingAdapter,
    get_route_matrix,
)

__all__ = [
    "GeoPoint",
    "GraphHopperClientAdapter",
    "OSRMClientAdapter",
    "PlanarCachedAdapter",
    "RouteMatrix",
    "RoutingAdapter",
    "get_route_matrix",
]
