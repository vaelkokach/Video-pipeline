from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List


@dataclass
class RuntimeState:
    latest_frame_index: int = 0
    latest_summary: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def publish_sync(self, payload: Dict[str, Any]) -> None:
        with self.lock:
            self.latest_frame_index = int(payload.get("frame_index", self.latest_frame_index))
            self.latest_summary = payload
            self.events.append(payload)
            if len(self.events) > 500:
                self.events = self.events[-500:]
