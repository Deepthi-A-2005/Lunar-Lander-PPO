"""Checkpoint management: best/latest models plus a JSON registry."""
from __future__ import annotations

import json
from pathlib import Path


class CheckpointManager:
    def __init__(self, save_dir="models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.save_dir / "registry.json"

    def latest_path(self) -> Path:
        return self.save_dir / "ppo_lander_latest.pt"

    def best_path(self) -> Path:
        return self.save_dir / "ppo_lander_best.pt"

    def save(self, agent, kind="latest", meta=None):
        path = self.latest_path() if kind == "latest" else self.best_path()
        agent.save_checkpoint(str(path), meta=meta or {})
        self._record(kind, path, meta or {})
        return path

    def _record(self, kind, path, meta):
        registry = []
        if self.registry_path.exists():
            try:
                registry = json.loads(self.registry_path.read_text())
            except json.JSONDecodeError:
                registry = []
        registry = [entry for entry in registry if entry.get("kind") != kind]
        registry.append({"kind": kind, "path": str(path), "meta": meta})
        self.registry_path.write_text(json.dumps(registry, indent=2))

    def registry(self):
        if not self.registry_path.exists():
            return []
        try:
            return json.loads(self.registry_path.read_text())
        except json.JSONDecodeError:
            return []
