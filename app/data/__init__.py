"""Data ingestion, validation, normalization, snapshots, and synthetic generation."""

from app.data.ingestion import (
    ingest_orders_file,
    publish_content_addressed_snapshot,
    verify_snapshot,
)
from app.data.normalization import (
    load_dataset_snapshot,
    normalize_orders_frame,
    save_dataset_snapshot,
)
from app.data.synthetic import generate_synthetic_dataset
from app.data.validation import (
    DatasetError,
    ValidationIssue,
    ValidationReport,
    validate_dataset,
    validate_raw_order_rows,
)

__all__ = [
    "DatasetError",
    "ValidationIssue",
    "ValidationReport",
    "generate_synthetic_dataset",
    "ingest_orders_file",
    "load_dataset_snapshot",
    "normalize_orders_frame",
    "publish_content_addressed_snapshot",
    "save_dataset_snapshot",
    "validate_dataset",
    "validate_raw_order_rows",
    "verify_snapshot",
]
