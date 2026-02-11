import json
import os
import re
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _normalize_sql(sql: str) -> str:
    return sql.strip()


def _has_nondeterminism(sql: str) -> bool:
    if not isinstance(sql, str):
        return False
    lowered = sql.lower()
    patterns = (
        'random(',
        "datetime('now')",
        'current_timestamp',
        "date('now')",
        "time('now')",
    )
    return any(p in lowered for p in patterns)


def _get_table_columns(db_path: str, table: str) -> Optional[List[str]]:
    if not db_path or not os.path.exists(db_path):
        return None
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2)
        try:
            cur = conn.cursor()
            cur.execute(f'PRAGMA table_info("{table}")')
            cols = [row[1] for row in cur.fetchall() if len(row) > 1]
            return cols or None
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        return None


def _try_expand_select_star(sql: str, db_path: Optional[str]) -> Tuple[str, Optional[str]]:
    if not isinstance(sql, str) or not sql.strip() or not db_path:
        return sql, None
    import re as _re
    text = sql.strip()
    if not _re.search(r"^\s*select\s+\*\b", text, flags=_re.IGNORECASE):
        return sql, None
    m = _re.match(r"^\s*select\s+\*\s+from\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:as\s+)?([A-Za-z_][A-Za-z0-9_]*))?(\s.*)?$",
                  text, flags=_re.IGNORECASE | _re.DOTALL)
    if not m:
        return sql, "expand_star_skip: complex SELECT or multiple tables"
    table, alias, tail = m.group(1), m.group(2), (m.group(3) or '')
    if _re.search(r"\bjoin\b", tail or '', flags=_re.IGNORECASE):
        return sql, "expand_star_skip: join detected"
    cols = _get_table_columns(db_path, table)
    if not cols:
        return sql, "expand_star_skip: no columns found"
    qual = alias or table
    col_list = ", ".join([f"{qual}.\"{c}\"" for c in cols])
    rewritten = f"SELECT {col_list} FROM {table}"
    if alias and alias.lower() != table.lower():
        rewritten += f" AS {alias}"
    if tail:
        rewritten += tail
    return rewritten, "expand_star_applied"


def _extract_all_sql_from_dialogue(dialogue_text: str) -> List[str]:
    """Extract all SQL queries from a dialogue text.

    Improvements over previous version:
    1. Handle COSQL style tool_call blocks where name may be 'sqlexe' and argument key is 'query'.
    2. Fallback to scanning raw lines starting with common SQL verbs (SELECT/UPDATE/DELETE/INSERT/WITH/CREATE) even if
       they were entered directly by the user, not inside tool_call JSON.
    3. Robustly parse nested JSON inside <tool_call> even if arguments field is itself a JSON string.
    4. De-duplicate while preserving original order.
    """
    if not isinstance(dialogue_text, str):
        return []
    tag_pattern = r"<tool[_ ]?call\b([^>]*)>(.*?)</tool[_ ]?call>"
    collected: List[str] = []

    # Parse tool_call/toolcall JSON blocks
    for match in re.finditer(tag_pattern, dialogue_text, flags=re.DOTALL | re.IGNORECASE):
        attrs = match.group(1) or ''
        payload = match.group(2) or ''

        attr_name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', attrs, flags=re.IGNORECASE)
        name_from_attr = (attr_name_match.group(1).strip().lower() if attr_name_match else '')

        payload_text = payload.strip()
        if not payload_text:
            continue

        data: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(payload_text)
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            json_match = re.search(r"\{.*\}", payload_text, flags=re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict):
                        data = parsed
                except Exception:
                    data = None
        if data is None:
            continue

        arguments = data.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass
        if not isinstance(arguments, dict):
            arguments = data  # fall back to payload body if arguments missing

        if not isinstance(arguments, dict):
            continue

        raw_name = data.get("name")
        name_value = ''
        if isinstance(raw_name, str):
            name_value = raw_name.strip().lower()
        elif name_from_attr:
            name_value = name_from_attr
        if not name_value and any(k in arguments for k in ("sql", "query")):
            name_value = "sql"
        if name_value not in ("sql", "sqlexe"):
            continue


        sql_field = arguments.get("sql")
        if not isinstance(sql_field, str):
            sql_field = arguments.get("query")
        if isinstance(sql_field, str):
            stripped = sql_field.strip()
            if stripped:
                collected.append(stripped)

    # New heuristic: Scan for JSON objects that might be tool calls (without tags)
    # This handles cases where the model outputs raw JSON like {"name": "sqlexe", "arguments": ...}
    try:
        # 1. Try parsing the whole text as JSON (single object or list)
        candidates = []
        try:
            parsed_full = json.loads(dialogue_text)
            if isinstance(parsed_full, list):
                candidates.extend(parsed_full)
            elif isinstance(parsed_full, dict):
                candidates.append(parsed_full)
        except json.JSONDecodeError:
            # 2. If full parse fails, try regex to find JSON-like blocks
            # We look for blocks containing "sqlexe" to reduce false positives
            if 'sqlexe' in dialogue_text.lower():
                # Simple regex for non-nested JSON objects
                for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"sqlexe"[^{}]*\}', dialogue_text, re.IGNORECASE | re.DOTALL):
                    try:
                        candidates.append(json.loads(match.group(0)))
                    except:
                        pass
                # Regex for objects with arguments (potentially nested, but we try a simple greedy match first)
                # This is imperfect but covers many cases.
                # A better approach is to find "sqlexe" and expand outwards, but that's complex.
                # Let's rely on the fact that if it's valid JSON, it often starts with {
                pass

        for item in candidates:
            if not isinstance(item, dict):
                continue
            
            # Check for OpenAI function call format: {"function": {"name": "sqlexe", ...}}
            func = item.get('function')
            if isinstance(func, dict):
                item = func
            
            name = item.get('name')
            if isinstance(name, str) and 'sqlexe' in name.lower():
                args = item.get('arguments')
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        pass
                if isinstance(args, dict):
                    sql = args.get('sql') or args.get('query')
                    if isinstance(sql, str) and sql.strip():
                        collected.append(sql.strip())
    except Exception:
        pass

    # Raw line heuristic: capture standalone SQL not wrapped in tool_call
    # Avoid capturing JSON lines (those start with '{' typically) and think tags.
    raw_lines = dialogue_text.splitlines()
    sql_prefixes = ("select", "update", "delete", "insert", "with", "create")
    for line in raw_lines:
        lstrip = line.lstrip()
        lowered_line = lstrip.lower()
        if not lstrip or lstrip.startswith('{') or '<tool' in lowered_line:
            continue
        for p in sql_prefixes:
            if lowered_line.startswith(p):
                # Trim trailing tool_response markers etc.
                candidate = lstrip.rstrip()
                # Remove trailing extraneous characters (common ';')
                candidate = candidate.rstrip(';').strip()
                if candidate and candidate not in collected:
                    collected.append(candidate)
                break

    # Deduplicate while preserving order (already prevented duplicates in raw heuristic, but ensure uniqueness overall)
    seen = set()
    unique: List[str] = []
    for q in collected:
        if q not in seen:
            unique.append(q)
            seen.add(q)
    return unique


def _extract_actual_sql_from_dialogue(dialogue_text: str) -> Optional[str]:
    queries = _extract_all_sql_from_dialogue(dialogue_text)
    if not queries:
        return None
    return queries[-1]


def _normalize_sql_for_compare(sql: str, *, collapse_ws: bool = True, strip_semicolon: bool = True) -> str:
    if not isinstance(sql, str):
        return ''
    s = sql.strip()
    if strip_semicolon and s.endswith(';'):
        s = s[:-1].rstrip()
    if collapse_ws:
        s = re.sub(r"\s+", " ", s)
    return s


def _extract_sql_from_obj(row_obj: Any) -> Optional[str]:
    try:
        if row_obj is None:
            return None
        if isinstance(row_obj, str):
            text = row_obj.strip()
            if not text:
                return None
            extracted = _extract_actual_sql_from_dialogue(text)
            if extracted:
                return extracted
            lowered = text.lstrip().lower()
            for prefix in ("select", "insert", "update", "delete", "with", "create"):
                if lowered.startswith(prefix):
                    return text
            return None
        if hasattr(row_obj, "to_dict"):
            return _extract_sql_from_obj(row_obj.to_dict())
        if isinstance(row_obj, dict):
            sql_field = row_obj.get("sql")
            if isinstance(sql_field, str) and sql_field.strip():
                return sql_field
            arguments = row_obj.get("arguments")
            if isinstance(arguments, dict):
                sql_field = arguments.get("sql")
                if isinstance(sql_field, str) and sql_field.strip():
                    return sql_field
            for key in ("reward_model", "ground_truth", "value"):
                nested = row_obj.get(key)
                if isinstance(nested, dict):
                    extracted = _extract_sql_from_obj(nested)
                    if extracted:
                        return extracted
            actions = row_obj.get("actions")
            if isinstance(actions, (list, tuple)):
                for action in actions:
                    extracted = _extract_sql_from_obj(action)
                    if extracted:
                        return extracted
        if isinstance(row_obj, (list, tuple)):
            for item in row_obj:
                extracted = _extract_sql_from_obj(item)
                if extracted:
                    return extracted
    except Exception:
        return None
    return None


def _sql_key(sql: Optional[str]) -> Optional[str]:
    if not sql:
        return None
    normalized = _normalize_sql(sql)
    if not normalized:
        return None
    import hashlib
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
