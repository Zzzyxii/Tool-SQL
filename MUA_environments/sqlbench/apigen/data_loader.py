# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loader for the SQLBench apigen (TauBench airline SQL) environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from MUA_environments.base.data_loader import BaseDataLoader


class SQLBenchApigenDataLoader(BaseDataLoader):
    """Resolve shared resources for the apigen SQL environment."""

    DEFAULT_DB_NAME = "taubench_airline"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._explicit_db_path = config.get("db_path")
        self._explicit_schema_path = config.get("schema_path")
        self._db_name = config.get("db_name", self.DEFAULT_DB_NAME)
        self._dataset_hint = self._infer_dataset_hint()
        self._root = self._resolve_root_with_fallbacks()

    def _resolve_root_with_fallbacks(self) -> Path:
        candidates = []
        for key in ("apigen_root", "db_root", "data_root"):
            raw = self.config.get(key)
            if raw:
                candidates.append(raw)
        env_root = os.getenv("APIGEN_DB_ROOT")
        if env_root:
            candidates.append(env_root)
        repo_root = Path(__file__).resolve().parents[3]
        workspace_root = repo_root.parent
        base = repo_root
        candidates.append(repo_root / "verl" / "tools" / "sqlbench_apigen" / "data")
        hydra_root = os.getenv("HYDRA_ORIG_CWD")

        def resolve_candidate(raw_path: Any) -> Path:
            p = Path(raw_path)
            if p.is_absolute():
                return p.resolve()
            if hydra_root and (Path(hydra_root) / p).exists():
                return (Path(hydra_root) / p).resolve()
            if (Path.cwd() / p).exists():
                return (Path.cwd() / p).resolve()
            if (repo_root / p).exists():
                return (repo_root / p).resolve()
            return (workspace_root / p).resolve()

        checked: list[Path] = []

        def prefer_dataset_root(resolved: Path) -> Optional[Path]:
            if not resolved.exists():
                return None
            hint = self._dataset_hint
            if hint:
                if resolved.name.lower() == hint:
                    return resolved
                candidate = resolved / hint
                if candidate.exists():
                    return candidate
                for sqlite_file in resolved.glob("*.sqlite"):
                    if hint in sqlite_file.name.lower():
                        return resolved
                return None
            if any(resolved.glob("*.sqlite")):
                return resolved
            return None

        for cand in candidates:
            resolved = resolve_candidate(cand)
            checked.append(resolved)
            preferred = prefer_dataset_root(resolved)
            if preferred is not None:
                return preferred.resolve()
        return checked[0] if checked else (repo_root / "verl" / "tools" / "sqlbench_apigen" / "data")

    def _infer_dataset_hint(self) -> Optional[str]:
        def _normalise(value: Optional[Any]) -> Optional[str]:
            if not value:
                return None
            text = str(value).lower()
            for key in ("airline", "retail"):
                if key in text:
                    return key
            return None

        candidates: Iterable[Any] = (
            self.config.get("dataset"),
            self.config.get("data_set"),
            self.config.get("trainset_version"),
            self.config.get("ability"),
            self.config.get("ability_hint"),
            self._db_name,
        )
        for value in candidates:
            hint = _normalise(value)
            if hint:
                return hint
        return None

    def _resolve_relative(self, raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p.resolve()
        hydra_root = os.getenv("HYDRA_ORIG_CWD")
        candidate_bases = []
        if hydra_root:
            candidate_bases.append(Path(hydra_root))
        candidate_bases.extend([self._root, Path.cwd()])
        for base in candidate_bases:
            candidate = (base / p).resolve()
            if candidate.exists():
                return candidate
        return (self._root / p).resolve()

    def _resolve_sqlite_path(self) -> Path:
        if self._explicit_db_path:
            return self._resolve_relative(str(self._explicit_db_path))
        for name in self._candidate_sqlite_names():
            candidate = self._root / name
            if candidate.exists():
                return candidate.resolve()
        candidates = sorted(self._root.glob("*.sqlite"))
        if candidates:
            return candidates[0].resolve()
        raise FileNotFoundError(
            f"Could not find TauBench apigen SQLite database under {self._root}"
        )

    def _candidate_sqlite_names(self) -> Iterable[str]:
        names = []
        hint = self._dataset_hint
        if hint == "airline":
            names.extend(["airline.sqlite", "taubench_airline.sqlite"])
        elif hint == "retail":
            names.extend(["retail.sqlite", "taubench_retail.sqlite"])
        else:
            names.extend([
                "taubench_airline.sqlite",
                "airline.sqlite",
                "taubench_retail.sqlite",
                "retail.sqlite",
            ])
        db_name = (self._db_name or "").strip().lower()
        if db_name:
            if db_name.endswith(".sqlite"):
                names.append(db_name)
            else:
                names.append(f"{db_name}.sqlite")
        # Preserve order but drop duplicates
        seen = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                yield name

    def _resolve_schema_path(self, sqlite_path: Path) -> Path:
        if self._explicit_schema_path:
            return self._resolve_relative(str(self._explicit_schema_path))
        preferred = sqlite_path.with_name("schema.sql")
        if preferred.exists():
            return preferred.resolve()
        fallback = self._root / "schema.sql"
        if fallback.exists():
            return fallback.resolve()
        raise FileNotFoundError(
            f"Could not locate schema.sql for TauBench apigen dataset under {self._root}"
        )

    def load_data(self) -> Dict[str, Any]:
        sqlite_path = self._resolve_sqlite_path()
        schema_path = self._resolve_schema_path(sqlite_path)
        return {
            "db_name": self._db_name,
            "db_path": str(sqlite_path),
            "schema_path": str(schema_path),
            "data_root": str(self._root),
        }

    def get_data_schema(self) -> Dict[str, Any]:
        return {
            "db_name": str,
            "db_path": str,
            "schema_path": str,
            "data_root": str,
        }
