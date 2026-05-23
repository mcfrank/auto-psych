"""Shared validation result type."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Validated:
    ok: bool
    message: str
    details: Optional[Dict[str, Any]] = None
