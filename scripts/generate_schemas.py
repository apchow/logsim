"""Generate public JSON Schemas from authoritative Pydantic v2 domain models."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from app.domain import entities, events


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    output.mkdir(parents=True, exist_ok=True)
    for mod in (entities, events):
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
                (output / f"{name}.schema.json").write_text(
                    json.dumps(obj.model_json_schema(), indent=2) + "\n",
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
