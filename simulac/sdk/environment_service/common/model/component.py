from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple


@dataclass
class RenderingComponent:
    texture_uri: Optional[str] = None
    source_uri: Optional[str] = None
    color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    scale: Tuple[float, float, float] = (1, 1, 1)

    # additional field for web rendering
    material_properties: Dict[str, Any] = field(default_factory=dict)
    asset_bundle: Dict[str, Any] = field(default_factory=dict)

