"""Engine/model/route descriptors parsed from config/engines.yaml."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _substitute(value: Any, env: dict[str, str]) -> Any:
    """Expand ${VAR} references in strings / lists / dicts using the given env."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            return env.get(key, os.environ.get(key, m.group(0)))
        return _VAR_RE.sub(repl, value)
    if isinstance(value, list):
        return [_substitute(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, env) for k, v in value.items()}
    return value


@dataclass
class EngineSpec:
    name: str
    kind: str = "binary"               # "binary" (exe + args) or "python" (module)
    binary: str | None = None
    module: str | None = None
    port: int = 8001
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    health_path: str = "/v1/models"
    ready_timeout_s: int = 300          # wait up to 5 min for big models to start
    kill_timeout_s: int = 30


@dataclass
class ModelSpec:
    name: str
    path: str | None = None
    path_ik: str | None = None
    size_gb: float | None = None


@dataclass
class Route:
    alias: str
    engine: str
    model: str
    ttl_s: int = 1200                  # 20 min default; big models override lower on small


@dataclass
class EngineConfig:
    engines: dict[str, EngineSpec]
    models: dict[str, ModelSpec]
    routes: dict[str, Route]
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "EngineConfig":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        env: dict[str, str] = dict(os.environ)
        # Project-local defaults so yaml can refer to paths without needing host envvars.
        env.setdefault("REPO_ROOT", str(p.resolve().parent.parent))
        env.setdefault("MODEL_DIR", env.get("MODEL_DIR", str(Path("models").resolve())))

        raw = _substitute(raw, env)

        engines: dict[str, EngineSpec] = {}
        for name, spec in (raw.get("engines") or {}).items():
            spec = spec or {}
            engines[name] = EngineSpec(
                name=name,
                kind=spec.get("kind", "binary"),
                binary=spec.get("binary"),
                module=spec.get("module"),
                port=int(spec.get("port", 8001)),
                args=list(spec.get("args") or []),
                env=dict(spec.get("env") or {}),
                health_path=spec.get("health_path", "/v1/models"),
                ready_timeout_s=int(spec.get("ready_timeout_s", 300)),
                kill_timeout_s=int(spec.get("kill_timeout_s", 30)),
            )

        models: dict[str, ModelSpec] = {}
        for name, spec in (raw.get("models") or {}).items():
            spec = spec or {}
            models[name] = ModelSpec(
                name=name,
                path=spec.get("path"),
                path_ik=spec.get("path_ik"),
                size_gb=spec.get("size_gb"),
            )

        routes: dict[str, Route] = {}
        for alias, spec in (raw.get("routes") or {}).items():
            spec = spec or {}
            routes[alias] = Route(
                alias=alias,
                engine=spec["engine"],
                model=spec["model"],
                ttl_s=int(spec.get("ttl_s", 1200)),
            )

        return cls(
            engines=engines,
            models=models,
            routes=routes,
            defaults=raw.get("defaults") or {},
        )

    def resolve_route(self, alias: str) -> tuple[EngineSpec, ModelSpec, Route]:
        if alias not in self.routes:
            raise KeyError(f"unknown route/alias '{alias}'. Known: {list(self.routes)}")
        r = self.routes[alias]
        if r.engine not in self.engines:
            raise KeyError(f"route '{alias}' points at unknown engine '{r.engine}'")
        if r.model not in self.models:
            raise KeyError(f"route '{alias}' points at unknown model '{r.model}'")
        return self.engines[r.engine], self.models[r.model], r

    def expand_engine_args(
        self, engine: EngineSpec, model: ModelSpec
    ) -> tuple[list[str], dict[str, str]]:
        """Substitute model-derived placeholders (${MODEL_PATH}, ${MODEL_PATH_IK})."""
        local_env: dict[str, str] = {}
        if model.path:
            local_env["MODEL_PATH"] = model.path
        if model.path_ik:
            local_env["MODEL_PATH_IK"] = model.path_ik
        local_env.setdefault("MODEL_PATH_IK", model.path or "")  # fallback
        args = [_substitute(a, local_env) for a in engine.args]
        env = {k: _substitute(v, local_env) for k, v in engine.env.items()}
        return args, env
