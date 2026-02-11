from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import sqlite3

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GetDBSchemaTool(BaseTool):
    """Return schema text for a database.

    Config:
      data_root: path to dataset root (will look under `database/<db_name>`)
      timeout: sqlite connection timeout
    Parameters during execute:
      query: db_name (for compatibility with provided schema spec)
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        raw_root = config.get("data_root") or config.get("db_root") or "./data"
        self._module_dir = Path(__file__).resolve().parent
        orig_cwd = os.getenv("HYDRA_ORIG_CWD")
        self._run_root = Path(orig_cwd).resolve() if orig_cwd else Path.cwd().resolve()
        self._root = self._resolve_relative_path(raw_root)
        self.timeout = config.get("timeout", 5)
        self._database_dir = (self._root / "database").resolve()
        if not self._database_dir.exists():
            logger.warning("COSQL database directory missing: %s", self._database_dir)
        self._instances: Dict[str, Dict[str, Any]] = {}

    def _resolve_relative_path(self, target: str) -> Path:
        p = Path(target)
        if p.is_absolute():
            return p.resolve()
        candidates = [(self._run_root / p).resolve(), (self._module_dir / p).resolve()]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _resolve_any_path(self, target: str) -> Path:
        """Resolve a path relative to workspace, data root, or module directory."""
        p = Path(target).expanduser()
        if p.is_absolute():
            return p.resolve()
        candidates = [
            (self._run_root / p).resolve(),
            (self._root / p).resolve(),
            (self._module_dir / p).resolve(),
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    def _coerce_path(self, value: Optional[Any]) -> Optional[Path]:
        if not value:
            return None
        if isinstance(value, Path):
            return value
        return self._resolve_any_path(str(value))

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            import uuid
            instance_id = str(uuid.uuid4())
        self._instances[instance_id] = {
            "executed": False,
            "error": "",
            "schema_path": self._coerce_path(kwargs.get("schema_path")),
            "db_path": self._coerce_path(kwargs.get("db_path")),
            "db_name": kwargs.get("db_name") or kwargs.get("database"),
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict[str, Any]]:
        inst = self._instances.get(instance_id)
        if not inst:
            await self.create(instance_id)
            inst = self._instances[instance_id]

        # Check for potential hallucination where agent tries to filter schema by table name
        query_arg = parameters.get("query")
        db_name_arg = kwargs.get("db_name") or kwargs.get("database") or inst.get("db_name")
        if query_arg and db_name_arg and query_arg != db_name_arg:
             logger.warning(f"getdbschema received query='{query_arg}' which differs from db_name='{db_name_arg}'. This tool returns the FULL schema regardless of the query argument. The agent might be hallucinating table filtering capabilities.")

        schema_path_override = parameters.get("schema_path") or kwargs.get("schema_path")
        db_path_override = parameters.get("db_path") or kwargs.get("db_path")
        db_name = parameters.get("query") or kwargs.get("db_name") or kwargs.get("database") or inst.get("db_name")

        schema_path = self._coerce_path(schema_path_override) if schema_path_override else inst.get("schema_path")
        db_path = self._coerce_path(db_path_override) if db_path_override else inst.get("db_path")
        if schema_path:
            inst["schema_path"] = schema_path
        if db_path:
            inst["db_path"] = db_path
        if db_name and db_name != inst.get("db_name"):
            inst["db_name"] = db_name

        def _maybe_strip(text: str) -> str:
            strip_inserts = bool(os.getenv("COSQL_STRIP_INSERTS", "1") == "1" or self.config.get("strip_inserts", True))
            return self._remove_insert_statements(text) if strip_inserts else text

        ddl_text: Optional[str] = None
        schema_source: Optional[str] = None

        if schema_path:
            if schema_path.exists():
                ddl_text = _maybe_strip(schema_path.read_text(encoding="utf-8"))
                schema_source = str(schema_path)
            else:
                logger.warning("Schema file not found at override path: %s", schema_path)

        sibling_schema: Optional[Path] = None
        if ddl_text is None and db_path:
            candidate = Path(db_path).with_name("schema.sql")
            if candidate.exists():
                sibling_schema = candidate
        if ddl_text is None and sibling_schema:
            ddl_text = _maybe_strip(sibling_schema.read_text(encoding="utf-8"))
            schema_source = str(sibling_schema)

        db_dir: Optional[Path] = None
        if db_name:
            candidate_dir = (self._database_dir / db_name).resolve()
            if candidate_dir.exists():
                db_dir = candidate_dir

        ddl_file = None
        if ddl_text is None and db_dir:
            for candidate_name in ("schema.sql", "TextBookExampleSchema.sql"):
                c = db_dir / candidate_name
                if c.exists():
                    ddl_file = c
                    break
            if ddl_file:
                try:
                    ddl_text = _maybe_strip(ddl_file.read_text(encoding="utf-8"))
                    schema_source = str(ddl_file)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed reading schema file %s: %s", ddl_file, exc)

        sqlite_path: Optional[Path] = None
        if db_path and Path(db_path).exists():
            sqlite_path = Path(db_path).resolve()
        elif db_dir:
            sqlite_candidates = sorted(db_dir.glob("*.sqlite"))
            if not sqlite_candidates:
                msg = f"No sqlite file under {db_dir}"
                inst.update(error=msg, executed=False)
                return msg, 0.0, {}
            sqlite_path = sqlite_candidates[0]

        if sqlite_path is None:
            msg = "Neither db_path nor db_name resolved to an existing sqlite database"
            logger.warning(msg)
            inst.update(error=msg, executed=False)
            return msg, 0.0, {}

        if ddl_text is None:
            ddl_text = self._introspect_sqlite(str(sqlite_path))
            schema_source = schema_source or "introspected"

        payload = {
            "db_name": db_name or sqlite_path.stem,
            "source": schema_source or "introspected",
            "schema": ddl_text,
        }
        payload["db_path"] = str(sqlite_path)
        inst.update(executed=True, error="")
        return json.dumps(payload, ensure_ascii=False), 1.0, {"length": len(ddl_text)}

    def _introspect_sqlite(self, db_path: str) -> str:
        conn = sqlite3.connect(db_path, timeout=self.timeout)
        try:
            cur = conn.cursor()
            # tables
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            lines: List[str] = []
            for t in sorted(tables):
                lines.append(f"-- Table: {t}")
                cur.execute(f"PRAGMA table_info('{t}')")
                cols = cur.fetchall()  # cid, name, type, notnull, dflt_value, pk
                for cid, name, col_type, notnull, dflt, pk in cols:
                    nn = "NOT NULL" if notnull else ""
                    pk_tag = "PRIMARY KEY" if pk else ""
                    default_tag = f"DEFAULT {dflt}" if dflt is not None else ""
                    col_line = " ".join(x for x in [name, col_type, nn, pk_tag, default_tag] if x)
                    lines.append(f"  {col_line}")
                # foreign keys
                cur.execute(f"PRAGMA foreign_key_list('{t}')")
                fks = cur.fetchall()
                for fk in fks:
                    # (id, seq, table, from, to, on_update, on_delete, match)
                    lines.append(f"  FOREIGN KEY({fk[3]}) REFERENCES {fk[2]}({fk[4]})")
                lines.append("")
            return "\n".join(lines)
        finally:
            conn.close()

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        inst = self._instances.get(instance_id)
        if not inst:
            return 0.0
        if inst.get("executed") and not inst.get("error"):
            return 1.0
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

    # ---------------- Helpers ----------------
    def _remove_insert_statements(self, text: str) -> str:
        import re
        lines = text.splitlines()
        out: list[str] = []
        skipping = False
        for line in lines:
            stripped = line.lstrip()
            if not skipping and re.match(r"(?i)^INSERT\b", stripped):
                skipping = True
                # if INSERT statement ends on same line (semicolon), stop skipping immediately
                if ";" in stripped:
                    skipping = False
                continue
            if skipping:
                if ";" in stripped:
                    skipping = False
                continue
            out.append(line)
        return "\n".join(out)
