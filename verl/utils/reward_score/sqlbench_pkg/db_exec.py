import os
import shutil
import sqlite3
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

from .hash_utils import _build_payload, _hash_payload, _build_error_payload


def _split_sql_statements(sql_text: str) -> List[str]:
    if not isinstance(sql_text, str):
        return []
    parts = [segment.strip() for segment in sql_text.strip().split(';')]
    return [segment for segment in parts if segment]


def _extract_target_table(sql: str) -> Optional[str]:
    if not isinstance(sql, str):
        return None
    s = sql.strip()
    m = None
    import re
    m = re.match(r"^\s*insert\s+into\s+([A-Za-z_][A-Za-z0-9_\.]*)(?:\s|\(|$)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^\s*update\s+([A-Za-z_][A-Za-z0-9_\.]*)(?:\s|$)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^\s*delete\s+from\s+([A-Za-z_][A-Za-z0-9_\.]*)(?:\s|$)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _execute_sql_on_copy(
    db_path: str,
    sql: str,
    max_rows: int = 1000,
    timeout: int = 5,
    *,
    verify_writes: bool = True,
    verify_rows_limit: int = 1000,
) -> Tuple[Optional[List[Tuple]], Optional[int], Optional[str]]:
    if not db_path:
        return None, None, "missing_db_path"
    if not os.path.exists(db_path):
        return None, None, f"db_path '{db_path}' not found"
    if not isinstance(sql, str) or not sql.strip():
        return None, None, "empty_sql"

    tmp_path = None
    conn = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy2(db_path, tmp_path)

        conn = sqlite3.connect(tmp_path, timeout=timeout)
        cur = conn.cursor()
        sql_clean = sql.rstrip().rstrip(';')
        cur.execute(sql_clean)
        if cur.description:
            rows = cur.fetchmany(max_rows)
            return rows, None, None
        conn.commit()
        rowcount = cur.rowcount if hasattr(cur, 'rowcount') else None

        if verify_writes:
            table = _extract_target_table(sql_clean)
            if table:
                if '.' in table:
                    table_ref = table
                else:
                    table_ref = f'"{table}"'
                try:
                    cur.execute(f"SELECT * FROM {table_ref} LIMIT {verify_rows_limit};")
                    rows = cur.fetchall()
                    return rows, None, None
                except Exception:
                    pass
        return None, rowcount, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, None, f"exec error: {e}\n{tb}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _hash_sql_result(
    db_path: str,
    sql: str,
    max_rows: int = 1000,
    timeout: int = 5,
    *,
    verify_writes: bool = True,
    verify_rows_limit: int = 1000,
) -> Dict[str, Any]:
    rows, rowcount, error = _execute_sql_on_copy(
        db_path,
        sql,
        max_rows=max_rows,
        timeout=timeout,
        verify_writes=verify_writes,
        verify_rows_limit=verify_rows_limit,
    )
    if error:
        payload = _build_error_payload(error)
        digest = _hash_payload(payload)
        return {'hash': None, 'payload': payload, 'error': error, 'hash_fallback': digest}

    payload = _build_payload(rows, rowcount)
    digest = _hash_payload(payload)
    return {'hash': digest, 'payload': payload, 'error': None}


def _open_temp_conn(original_db_path: str, *, timeout: int = 5) -> Tuple[Optional[str], Optional[sqlite3.Connection], Optional[str]]:
    if not original_db_path or not os.path.exists(original_db_path):
        return None, None, "missing_db_path"
    try:
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy2(original_db_path, tmp_path)
        conn = sqlite3.connect(tmp_path, timeout=timeout)
        return tmp_path, conn, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, None, f"open_temp_conn error: {e}\n{tb}"


def _exec_and_hash_on_conn(
    conn: sqlite3.Connection,
    sql_text: str,
    *,
    max_rows: int = 1000,
    verify_writes: bool = True,
    verify_rows_limit: int = 1000,
) -> Dict[str, Any]:
    def _run_single(statement: str, *, allow_verify: bool) -> Dict[str, Any]:
        cur = conn.cursor()
        sql_clean = statement.rstrip().rstrip(';')
        cur.execute(sql_clean)
        if cur.description:
            rows = cur.fetchmany(max_rows)
            payload = _build_payload(rows, None)
            return {'hash': _hash_payload(payload), 'payload': payload, 'error': None}

        conn.commit()
        rowcount = cur.rowcount if hasattr(cur, 'rowcount') else None
        if allow_verify:
            table = _extract_target_table(sql_clean)
            if table:
                try:
                    table_ref = table if '.' in table else f'"{table}"'
                    cur.execute(f"SELECT * FROM {table_ref} LIMIT {verify_rows_limit};")
                    rows = cur.fetchall()
                    payload = _build_payload(rows, None)
                    return {'hash': _hash_payload(payload), 'payload': payload, 'error': None}
                except Exception:
                    pass
        payload = _build_payload(None, rowcount)
        return {'hash': _hash_payload(payload), 'payload': payload, 'error': None}

    try:
        statements = _split_sql_statements(sql_text)
        if not statements:
            payload = _build_payload([], None)
            return {'hash': _hash_payload(payload), 'payload': payload, 'error': None}

        last_result: Optional[Dict[str, Any]] = None
        for idx, stmt in enumerate(statements):
            allow_verify = verify_writes and (idx == len(statements) - 1)
            last_result = _run_single(stmt, allow_verify=allow_verify)
            if last_result.get('error'):
                return last_result
        return last_result or {'hash': None, 'payload': _build_payload([], None), 'error': None}
    except Exception as e:
        tb = traceback.format_exc()
        payload = _build_error_payload(f"exec error: {e}\n{tb}")
        return {'hash': None, 'payload': payload, 'error': str(e), 'hash_fallback': _hash_payload(payload)}
