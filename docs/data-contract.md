# Data Contracts & Snapshot Provenance

This specification integrates the fail-closed data contracts from `dep` and multi-table ingestion adapters from `Kimi_Agent_v` into the reverse-logistics simulation platform.

## Ingestion Rules

1. **String Identifier Preservation**: CSV ingestion reads identifiers with `dtype=str, keep_default_na=False` so values like `"0001"` and `"NA"` remain literal strings.
2. **Duplicate Column Rejection**: Both raw CSV headers and stripped headers are checked for duplicates prior to normalization.
3. **Mixed Provenance Guard**: Datasets mixing `synthetic=True` and `synthetic=False` rows fail closed with `ValueError`.
4. **UTC-Aware Timestamps**: ISO-8601 timestamps (`created_at`, `available_at`, `promise_at`, `promise_date`) must be timezone-aware, normalize to UTC, satisfy `created_at <= available_at <= promise_at`, and convert deterministically to elapsed simulation minutes.
5. **Atomic Content-Addressed Snapshots**: Every ingested dataset publishes an immutable snapshot directory under `<output>/snapshots/<snapshot_id>/` containing `orders.parquet`, `orders.jsonl`, and `manifest.json` (`source_sha256`, `canonical_sha256`, `row_count`, `synthetic`) verified via `verify_snapshot(...)`.
6. **Cryptographic Leakage Audit**: Every `PolicyDecision` records a SHA-256 `observation_hash` over the causal `DecisionContext` visible at decision timestamp `t`.

Full machine-readable JSON Schemas are generated into `docs/schemas/*.schema.json` via `python scripts/generate_schemas.py`.
