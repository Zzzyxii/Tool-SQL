import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _canonicalize_rows(rows: Iterable[Tuple]) -> List[List[Any]]:
    normalized: List[List[Any]] = []
    for row in rows:
        normalized.append([_normalize_cell(cell) for cell in row])
    # Sort with a key that handles None values safely
    normalized.sort(key=lambda x: tuple((0, cell) if cell is not None else (1, None) for cell in x))
    return normalized


def _hash_payload(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _build_payload(rows: Optional[List[Tuple]], rowcount: Optional[int]) -> Dict[str, Any]:
    if rows is not None:
        return {'type': 'rows', 'rows': _canonicalize_rows(rows)}
    return {'type': 'rowcount', 'count': rowcount}


def _build_error_payload(error: str) -> Dict[str, Any]:
    return {'type': 'error', 'error': error}


def _extract_result_hash_map(obj: Any) -> Dict[str, str]:
    if not isinstance(obj, dict):
        return {}
    hashes = obj.get('result_hashes')
    if isinstance(hashes, list):
        mapping: Dict[str, str] = {}
        for entry in hashes:
            if not isinstance(entry, dict):
                continue
            key = entry.get('sql_key')
            value = entry.get('hash')
            if key and value:
                mapping[str(key)] = str(value)
        return mapping
    for key in ('ground_truth', 'reward_model', 'value'):
        nested = obj.get(key)
        mapping = _extract_result_hash_map(nested)
        if mapping:
            return mapping
    return {}
