from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SQLExeTool(BaseTool):
    """Execute SQL 
    Config keys:
      db_root: path to COSQL dataset root (expects subfolder `database`).
      timeout: sqlite connection timeout.
    Dynamic kwargs (create/execute): db_name, db_path (override), db_file (if needed).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.timeout = config.get("timeout", 5)
        self.max_rows = config.get("max_rows", 100)
        self._instance: Dict[str, Dict[str, Any]] = {}
        self._module_dir = Path(__file__).resolve().parent
        orig_cwd = os.getenv("HYDRA_ORIG_CWD")
        self._run_root = Path(orig_cwd).resolve() if orig_cwd else Path.cwd().resolve()

        raw_root = config.get("db_root") or config.get("data_root") or "./data"
        self._db_root = self._resolve_relative_path(raw_root)
        # In COSQL layout actual DBs live in `<root>/database/<db_name>`
        self._database_dir = (self._db_root / "database").resolve()
        if not self._database_dir.exists():
            logger.warning("COSQL database directory not found: %s", self._database_dir)

    # ---------------- Path Resolution ----------------
    def _resolve_relative_path(self, target: str) -> Path:
        p = Path(target)
        if p.is_absolute():
            return p.resolve()
        candidates = [(self._run_root / p).resolve(), (self._module_dir / p).resolve()]
        for c in candidates:
            if c.exists():
                return c
        # fallback first candidate even if missing (later validated)
        return candidates[0]

    def _resolve_db_path(
        self,
        *,
        db_path: Optional[str] = None,
        db_name: Optional[str] = None,
        db_file: Optional[str] = None,
    ) -> str:
        if db_path:
            candidate = Path(db_path)
            if not candidate.is_absolute():
                candidate = (self._resolve_relative_path(db_path)).resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"Explicit db_path does not exist: {candidate}")
            return str(candidate)
        db_dir = (self._database_dir / db_name).resolve()
        
        if db_file:
            cand = db_dir / db_file
            if not cand.exists():
                raise FileNotFoundError(f"db_file '{db_file}' not found in {db_dir}")
            return str(cand.resolve())
        sqlite_candidates = sorted(db_dir.glob("*.sqlite"))
        if not sqlite_candidates:
            raise FileNotFoundError(f"No .sqlite file in {db_dir}")
        if len(sqlite_candidates) > 1:
            names = ", ".join(c.name for c in sqlite_candidates)
            raise FileExistsError(f"Multiple .sqlite files; specify db_file. Found: {names}")
        return str(sqlite_candidates[0].resolve())

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        db_path = self._resolve_db_path(
            db_path=kwargs.get("db_path"),
            db_name=kwargs.get("db_name") or kwargs.get("database"),
            db_file=kwargs.get("db_file"),
        )
        # validate
        try:
            conn = sqlite3.connect(db_path, timeout=self.timeout)
            conn.close()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"SQLite connect failed: {db_path}: {exc}") from exc
        self._instance[instance_id] = {
            "db_path": db_path,
            "query": "",
            "result": None,
            "error": "",
            "executed": False,
            "local_copy": None,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict[str, Any]]:
        inst = self._instance.get(instance_id)
        if not inst:
            await self.create(instance_id, **kwargs)
            inst = self._instance[instance_id]
        query = parameters.get("query", "")
        if not isinstance(query, str):
            query = str(query)
        inst["query"] = query

        # allow override db_name on execute
        override = kwargs.get("db_name") or kwargs.get("database")
        if override:
            inst["db_path"] = self._resolve_db_path(db_name=override)

        use_path = inst["db_path"]
        # prepare local copy for potential writes
        if use_path.endswith(".sqlite"):
            if not inst.get("local_copy"):
                tmp_base = Path(tempfile.gettempdir()) / "verl_cosql_sql"
                tmp_base.mkdir(parents=True, exist_ok=True)
                local = tmp_base / f"{Path(use_path).stem}_{instance_id}.sqlite"
                if not local.exists():
                    shutil.copy(use_path, local)
                inst["local_copy"] = str(local)
            use_path = inst["local_copy"]

        try:
            result_obj = self._run_sql(query, use_path)
            inst.update({"result": result_obj, "error": "", "executed": True})
            # Handle list result for multi-statement queries
            row_count = 0
            if isinstance(result_obj, list):
                for res in result_obj:
                    row_count += len(res.get("rows", []))
            else:
                row_count = len(result_obj.get("rows", []))
            return json.dumps(result_obj, ensure_ascii=False), 1.0, {"rows": row_count}
        except Exception as exc:  # pylint: disable=broad-except
            err = f"SQL execution failed: {exc}"
            logger.warning(err)
            inst.update({"error": err, "executed": False})
            return err, 0.0, {}

    def _run_sql(self, query: str, db_path: str) -> Any:
        def _split_statements(sql: str) -> List[str]:
            return [segment.strip() for segment in sql.split(";") if segment.strip()]

        cleaned_query = query.strip()
        if not cleaned_query:
            return {"columns": [], "rows": [], "status": "EMPTY"}

        conn = sqlite3.connect(db_path, timeout=self.timeout)
        try:
            statements = _split_statements(cleaned_query)
            if len(statements) > 1:
                results = []
                cur = conn.cursor()
                for stmt in statements:
                    try:
                        cur.execute(stmt)
                        if cur.description:
                            rows = cur.fetchmany(self.max_rows + 1)
                            truncated = False
                            if len(rows) > self.max_rows:
                                rows = rows[:self.max_rows]
                                truncated = True
                            cols = [d[0] for d in cur.description]
                            res = {"columns": cols, "rows": rows}
                            if truncated:
                                res["warning"] = f"Result truncated to {self.max_rows} rows."
                            results.append(res)
                        else:
                            results.append({"columns": [], "rows": [], "status": f"OK {cur.rowcount} rows"})
                    except Exception as e:
                        results.append({"error": str(e)})
                conn.commit()
                return results

            cur = conn.cursor()
            cur.execute(cleaned_query)
            # Check if the query returned any data (e.g. SELECT, WITH ... SELECT, RETURNING)
            if cur.description:
                rows = cur.fetchmany(self.max_rows + 1)
                truncated = False
                if len(rows) > self.max_rows:
                    rows = rows[:self.max_rows]
                    truncated = True
                cols = [d[0] for d in cur.description]
                res = {"columns": cols, "rows": rows}
                if truncated:
                    res["warning"] = f"Result truncated to {self.max_rows} rows."
                return res
            conn.commit()
            return {"columns": [], "rows": [], "status": f"OK {cur.rowcount} rows"}
        finally:
            conn.close()

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        inst = self._instance.get(instance_id)
        if not inst:
            return 0.0
        if inst.get("executed") and not inst.get("error"):
            return 0.0  # reward shaping left to external scorer
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance.pop(instance_id, None)
