"""Deterministic, snapshot-cached routing adapters (Planar, OSRM, GraphHopper)."""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field


class GeoPoint(BaseModel):
    """Named coordinate point supporting planar (x_km, y_km) or WGS84 (lat, lon)."""

    location_id: str
    x_km: float = 0.0
    y_km: float = 0.0
    lat: float | None = None
    lon: float | None = None


class RouteMatrix(BaseModel):
    """Cached origin-destination distance and duration matrix with provenance hash."""

    provider: str
    profile: str
    api_version: str
    departure_minute_bucket: int = 0
    cache_hit: bool = False
    origins: list[str]
    destinations: list[str]
    distances_km: list[list[float]]
    durations_minutes: list[list[float]]
    matrix_sha256: str
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


def _compute_cache_key(
    provider: str,
    api_version: str,
    profile: str,
    departure_bucket: int,
    origins: list[GeoPoint],
    destinations: list[GeoPoint],
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "api_version": api_version,
            "profile": profile,
            "departure_bucket": departure_bucket,
            "origins": [o.model_dump(mode="json") for o in origins],
            "destinations": [d.model_dump(mode="json") for d in destinations],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _planar_distance_and_duration(
    a: GeoPoint, b: GeoPoint, speed_kmph: float = 42.0, circuity: float = 1.18
) -> tuple[float, float]:
    if a.location_id == b.location_id:
        return 0.0, 0.0
    if a.lat is not None and a.lon is not None and b.lat is not None and b.lon is not None:
        # Equirectangular local approximation in km
        d_lat = (b.lat - a.lat) * 111.0
        d_lon = (b.lon - a.lon) * 111.0 * math.cos(math.radians((a.lat + b.lat) / 2.0))
        dist = round(max(0.5, math.hypot(d_lat, d_lon) * circuity), 3)
    else:
        dist = round(max(0.5, math.hypot(a.x_km - b.x_km, a.y_km - b.y_km) * circuity), 3)
    dur = round(max(2.0, (dist / max(10.0, speed_kmph)) * 60.0), 2)
    return dist, dur


class RoutingAdapter(ABC):
    """Deterministic routing adapter interface with persistent matrix caching."""

    provider_name: str
    api_version: str

    def __init__(self, cache_path: str | Path = "artifacts/cache/route_cache.json") -> None:
        self.cache_path = Path(cache_path)

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self, cache: dict[str, dict[str, Any]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    @abstractmethod
    def _fetch_matrix(
        self,
        origins: list[GeoPoint],
        destinations: list[GeoPoint],
        profile: str,
    ) -> tuple[list[list[float]], list[list[float]], dict[str, Any]]:
        """Return (distances_km, durations_minutes, raw_metadata)."""

    def get_route_matrix(
        self,
        origins: list[GeoPoint],
        destinations: list[GeoPoint],
        profile: str = "driving",
        departure_minute: float = 0.0,
    ) -> RouteMatrix:
        """Compute or replay a cached route matrix keyed by endpoint/profile/departure bucket."""
        bucket = int(departure_minute // 60.0)
        key = _compute_cache_key(
            self.provider_name, self.api_version, profile, bucket, origins, destinations
        )
        cache = self._load_cache()
        if key in cache:
            entry = dict(cache[key])
            entry["cache_hit"] = True
            return RouteMatrix.model_validate(entry)

        distances, durations, meta = self._fetch_matrix(origins, destinations, profile)
        matrix_sha = hashlib.sha256(
            json.dumps({"distances": distances, "durations": durations}, sort_keys=True).encode("utf-8")
        ).hexdigest()

        matrix = RouteMatrix(
            provider=self.provider_name,
            profile=profile,
            api_version=self.api_version,
            departure_minute_bucket=bucket,
            cache_hit=False,
            origins=[o.location_id for o in origins],
            destinations=[d.location_id for d in destinations],
            distances_km=distances,
            durations_minutes=durations,
            matrix_sha256=matrix_sha,
            raw_metadata=meta,
        )
        cache[key] = matrix.model_dump(mode="json")
        self._save_cache(cache)
        return matrix


class PlanarCachedAdapter(RoutingAdapter):
    """Offline deterministic Euclidean+circuity matrix adapter."""

    provider_name = "planar_cached"
    api_version = "1.0"

    def _fetch_matrix(
        self,
        origins: list[GeoPoint],
        destinations: list[GeoPoint],
        profile: str,
    ) -> tuple[list[list[float]], list[list[float]], dict[str, Any]]:
        speed = 38.0 if profile == "truck" else 42.0
        distances: list[list[float]] = []
        durations: list[list[float]] = []
        for orig in origins:
            d_row: list[float] = []
            t_row: list[float] = []
            for dest in destinations:
                dist, dur = _planar_distance_and_duration(orig, dest, speed_kmph=speed)
                d_row.append(dist)
                t_row.append(dur)
            distances.append(d_row)
            durations.append(t_row)
        return distances, durations, {"source": "deterministic_planar", "speed_kmph": speed}


class OSRMClientAdapter(RoutingAdapter):
    """OSRM Table API adapter with local cache and deterministic offline fallback."""

    provider_name = "osrm"
    api_version = "v1"

    def __init__(
        self,
        base_url: str | None = None,
        cache_path: str | Path = "artifacts/cache/route_cache.json",
        timeout_sec: float = 4.0,
    ) -> None:
        super().__init__(cache_path=cache_path)
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout_sec = timeout_sec

    def _fetch_matrix(
        self,
        origins: list[GeoPoint],
        destinations: list[GeoPoint],
        profile: str,
    ) -> tuple[list[list[float]], list[list[float]], dict[str, Any]]:
        all_have_geo = all(p.lat is not None and p.lon is not None for p in origins + destinations)
        if self.base_url and all_have_geo:
            try:
                pts = origins + destinations
                coords = ";".join(f"{p.lon},{p.lat}" for p in pts)
                src_idx = ";".join(str(i) for i in range(len(origins)))
                dst_idx = ";".join(str(len(origins) + j) for j in range(len(destinations)))
                url = f"{self.base_url}/table/v1/{profile}/{coords}"
                resp = httpx.get(
                    url,
                    params={
                        "sources": src_idx,
                        "destinations": dst_idx,
                        "annotations": "distance,duration",
                    },
                    timeout=self.timeout_sec,
                )
                if resp.status_code == 200:
                    body = resp.json()
                    if "distances" in body and "durations" in body:
                        dists = [[round(float(x) / 1000.0, 3) for x in r] for r in body["distances"]]
                        durs = [[round(float(x) / 60.0, 2) for x in r] for r in body["durations"]]
                        return dists, durs, {"source": "osrm_http", "base_url": self.base_url}
            except Exception as exc:
                fallback_dists, fallback_durs, _ = PlanarCachedAdapter(self.cache_path)._fetch_matrix(
                    origins, destinations, profile
                )
                return fallback_dists, fallback_durs, {
                    "source": "osrm_offline_fallback",
                    "error": type(exc).__name__,
                }

        dists, durs, meta = PlanarCachedAdapter(self.cache_path)._fetch_matrix(
            origins, destinations, profile
        )
        return dists, durs, {**meta, "adapter": "osrm_fallback"}


class GraphHopperClientAdapter(RoutingAdapter):
    """GraphHopper Matrix API adapter with local cache and deterministic offline fallback."""

    provider_name = "graphhopper"
    api_version = "v1"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_path: str | Path = "artifacts/cache/route_cache.json",
        timeout_sec: float = 4.0,
    ) -> None:
        super().__init__(cache_path=cache_path)
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def _fetch_matrix(
        self,
        origins: list[GeoPoint],
        destinations: list[GeoPoint],
        profile: str,
    ) -> tuple[list[list[float]], list[list[float]], dict[str, Any]]:
        all_have_geo = all(p.lat is not None and p.lon is not None for p in origins + destinations)
        if self.base_url and all_have_geo:
            try:
                body_req: dict[str, Any] = {
                    "from_points": [[p.lon, p.lat] for p in origins],
                    "to_points": [[p.lon, p.lat] for p in destinations],
                    "out_arrays": ["distances", "times"],
                    "vehicle": profile,
                }
                params = {"key": self.api_key} if self.api_key else {}
                resp = httpx.post(
                    f"{self.base_url}/matrix",
                    json=body_req,
                    params=params,
                    timeout=self.timeout_sec,
                )
                if resp.status_code == 200:
                    body = resp.json()
                    if "distances" in body and "times" in body:
                        dists = [[round(float(x) / 1000.0, 3) for x in r] for r in body["distances"]]
                        durs = [[round(float(x) / 60000.0, 2) for x in r] for r in body["times"]]
                        return dists, durs, {"source": "graphhopper_http", "base_url": self.base_url}
            except Exception as exc:
                fallback_dists, fallback_durs, _ = PlanarCachedAdapter(self.cache_path)._fetch_matrix(
                    origins, destinations, profile
                )
                return fallback_dists, fallback_durs, {
                    "source": "graphhopper_offline_fallback",
                    "error": type(exc).__name__,
                }

        dists, durs, meta = PlanarCachedAdapter(self.cache_path)._fetch_matrix(
            origins, destinations, profile
        )
        return dists, durs, {**meta, "adapter": "graphhopper_fallback"}


def get_route_matrix(
    origins: list[GeoPoint],
    destinations: list[GeoPoint],
    provider: str = "osrm",
    profile: str = "driving",
    departure_minute: float = 0.0,
    base_url: str | None = None,
    cache_path: str | Path = "artifacts/cache/route_cache.json",
) -> RouteMatrix:
    """Resolve and execute a cached deterministic routing adapter."""
    prov = provider.lower()
    adapter: RoutingAdapter
    if prov == "osrm":
        adapter = OSRMClientAdapter(base_url=base_url, cache_path=cache_path)
    elif prov == "graphhopper":
        adapter = GraphHopperClientAdapter(base_url=base_url, cache_path=cache_path)
    else:
        adapter = PlanarCachedAdapter(cache_path=cache_path)
    return adapter.get_route_matrix(
        origins=origins,
        destinations=destinations,
        profile=profile,
        departure_minute=departure_minute,
    )
