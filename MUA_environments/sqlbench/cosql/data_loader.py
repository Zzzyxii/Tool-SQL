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

from pathlib import Path
from typing import Any, Dict, List, Optional
import os

from MUA_environments.base.data_loader import BaseDataLoader


class SQLBenchCoSQLDataLoader(BaseDataLoader):
    """Minimal data loader for COSQL environment.

    Unlike SQLBench single-db environments, COSQL contains multiple databases under
    `<db_root>/database/<db_name>/<db_name>.sqlite`. The execution tool will dynamically
    resolve the correct DB based on the tool call's `db_name` argument, so here we only
    validate and expose the shared `db_root`.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db_root = self._resolve_root_with_fallbacks()
        database_dir = self.db_root / "database"
        if not database_dir.exists():
            raise FileNotFoundError(
                "COSQL database directory missing under root after fallbacks checked: "
                f"{database_dir}"
            )

    def _resolve_root_with_fallbacks(self) -> Path:
        """Resolve COSQL root with multiple fallbacks.

        Priority order:
        1. Explicit config keys: cosql_db_root, db_root, data_root
        2. Environment variable COSQL_DB_ROOT
        3. Known absolute path from original tool config (/m2/.../MUA-RL/verl/tools/sqlbench_cosql/data)
        4. Current repo typical locations (./verl/tools/sqlbench_cosql/data)
        5. Ancestor-based resolution relative to this file
        Returns first path containing a 'database' subdirectory.
        """
        candidates: List[str] = []
        for key in ("cosql_db_root", "db_root", "data_root"):
            raw = self.config.get(key)
            if raw:
                candidates.append(str(raw))
        env_root = os.getenv("COSQL_DB_ROOT")
        if env_root:
            candidates.append(env_root)
        # Relative repo guesses
        candidates.extend([
            "verl/tools/sqlbench_cosql/data",
            "./verl/tools/sqlbench_cosql/data",
        ])
        # Path relative to this file (parents[4] => project root heuristic)
        base = Path(__file__).resolve().parents[4]
        candidates.append(str(base / "verl" / "tools" / "sqlbench_cosql" / "data"))

        checked: List[str] = []
        for cand in candidates:
            p = Path(cand)
            if not p.is_absolute():
                # Try HYDRA_ORIG_CWD then CWD then file-base
                hydra_root = os.getenv("HYDRA_ORIG_CWD")
                if hydra_root and (Path(hydra_root) / p).exists():
                    p = (Path(hydra_root) / p).resolve()
                elif (Path.cwd() / p).exists():
                    p = (Path.cwd() / p).resolve()
                else:
                    p = (base / p).resolve()
            checked.append(str(p))
            if (p / "database").exists():
                return p
        # Fall back to first candidate (even if invalid) so caller can raise informative error
        return Path(checked[0]) if checked else Path.cwd() / "data"

    def load_data(self) -> Dict[str, Any]:
        # Provide db_root so tools can derive per-db paths. Optionally could list DB names.
        database_dir = self.db_root / "database"
        if database_dir.exists():
            db_names = [d.name for d in database_dir.iterdir() if d.is_dir()]
        else:
            db_names = []
        return {"db_root": str(self.db_root), "db_names": db_names}

    def get_data_schema(self) -> Dict[str, Any]:
        return {"db_root": str, "db_names": list}
