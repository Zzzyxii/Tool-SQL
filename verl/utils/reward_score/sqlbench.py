# full code will be release after the paper publiced
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import asyncio

import sys
_this_dir = os.path.dirname(__file__)
_sqlbench_pkg_parent = _this_dir
if _sqlbench_pkg_parent not in sys.path:
    sys.path.insert(0, _sqlbench_pkg_parent)

from MUA_environments.manager import environment_manager
import torch
import numpy as np
from openai import OpenAI, AsyncOpenAI

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_PURE_SELECT_MAX_ROWS_DEFAULT = 100
_VALID_ACTION_NAMES = {'sqlexe'}
_SEMANTIC_MODEL_NAME = 'Qwen3-32B'
_SEMANTIC_BASE_URL = ''

# Simple top-level imports; reward loader ensures this directory is on sys.path
from sqlbench_pkg.sql_utils import (
    _normalize_sql as _normalize_sql,
    _extract_all_sql_from_dialogue as _extract_all_sql_from_dialogue,
    _normalize_sql_for_compare as _normalize_sql_for_compare,
    _extract_sql_from_obj as _extract_sql_from_obj,
    _sql_key as _sql_key,
)

from sqlbench_pkg import (
    _normalize_cell,
    _extract_result_hash_map,
    _hash_sql_result,
    _open_temp_conn,
    _exec_and_hash_on_conn,
)

from sqlbench_pkg.db_exec import _split_sql_statements

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


_TOOL_CALL_TAG_PATTERN = re.compile(r"<tool[_ ]?call\\b([^>]*)>(.*?)</tool[_ ]?call>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SIMPLE_SQLEXE_PATTERN = re.compile(r'"(tool_call|name|tool)"\s*:\s*"sqlexe"', re.IGNORECASE)


def _dict_contains_sqlexe_reference(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key in ('name', 'tool', 'tool_call'):
            value = obj.get(key)
            if isinstance(value, str) and value.strip().lower() == 'sqlexe':
                return True
            if isinstance(value, dict) and _dict_contains_sqlexe_reference(value):
                return True
        arguments = obj.get('arguments')
        if isinstance(arguments, (dict, list)) and _dict_contains_sqlexe_reference(arguments):
            return True
    if isinstance(obj, list):
        return any(_dict_contains_sqlexe_reference(item) for item in obj)
    return False


def _payload_mentions_sqlexe(payload: str) -> bool:
    snippet = payload.strip()
    if not snippet:
        return False
    candidates: List[Any] = []
    try:
        candidates.append(json.loads(snippet))
    except Exception:
        start = snippet.find('{')
        end = snippet.rfind('}')
        if start != -1 and end != -1 and end > start:
            body = snippet[start:end + 1]
            try:
                candidates.append(json.loads(body))
            except Exception:
                pass
    for candidate in candidates:
        if _dict_contains_sqlexe_reference(candidate):
            return True
    return False


def _text_has_sqlexe_tool_call(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value
    if 'sqlexe' not in text.lower():
        return False
    for match in _TOOL_CALL_TAG_PATTERN.finditer(text):
        attrs = match.group(1) or ''
        if 'sqlexe' in attrs.lower():
            return True
        payload = match.group(2) or ''
        if _payload_mentions_sqlexe(payload):
            return True
    for match in _CODE_FENCE_PATTERN.finditer(text):
        payload = match.group(1) or ''
        if _payload_mentions_sqlexe(payload):
            return True
    if _SIMPLE_SQLEXE_PATTERN.search(text):
        return True
    return False


def _payloads_to_text(payloads: List[Optional[Dict[str, Any]]]) -> str:
    rows_texts: List[str] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if payload.get('type') == 'rows':
            columns = payload.get('columns')
            rows = payload.get('rows')
            if isinstance(rows, list):
                rows_repr: List[Any] = []
                limit_rows = rows[:10]
                if isinstance(columns, list):
                    for row in limit_rows:
                        if isinstance(row, (list, tuple)):
                            rows_repr.append(dict(zip(columns, row)))
                        else:
                            rows_repr.append(row)
                    rows_texts.append(json.dumps({'columns': columns, 'rows': rows_repr}, ensure_ascii=False))
                else:
                    for row in limit_rows:
                        rows_repr.append(row)
                    rows_texts.append(json.dumps({'rows': rows_repr}, ensure_ascii=False))
        elif 'text' in payload:
            rows_texts.append(str(payload['text']))
    if not rows_texts:
        return ''
    return "\n".join(rows_texts)


def _convert_nested_to_lists(value: Any) -> Any:
    if isinstance(value, np.ndarray):  # type: ignore[attr-defined]
        return [_convert_nested_to_lists(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_convert_nested_to_lists(item) for item in value]
    if isinstance(value, tuple):
        return [_convert_nested_to_lists(item) for item in value]
    return value


def _normalize_output_payload(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):  # type: ignore[attr-defined]
        return _normalize_output_payload(value.tolist())
    if isinstance(value, dict):
        if 'type' in value:
            payload: Dict[str, Any] = {}
            for k, v in value.items():
                payload[k] = _convert_nested_to_lists(v)
            return payload
        if 'rows' in value:
            out: Dict[str, Any] = {
                'type': 'rows',
                'rows': _convert_nested_to_lists(value.get('rows')),
            }
            if 'columns' in value:
                out['columns'] = _convert_nested_to_lists(value.get('columns'))
            return out
        if 'count' in value:
            return {'type': 'rowcount', 'count': _convert_nested_to_lists(value.get('count'))}
        return {'type': 'json', 'value': _convert_nested_to_lists(value)}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {'type': 'text', 'text': text}
        return _normalize_output_payload(parsed)
    if isinstance(value, list):
        return {'type': 'rows', 'rows': _convert_nested_to_lists(value)}
    if isinstance(value, tuple):
        return {'type': 'rows', 'rows': [_convert_nested_to_lists(list(value))]}
    return {'type': 'value', 'value': _convert_nested_to_lists(value)}


def _outputs_to_text(outputs: List[Optional[Dict[str, Any]]]) -> str:
    if not outputs:
        return ''
    parts: List[str] = []
    for payload in outputs:
        if payload is None:
            continue
        if isinstance(payload, dict):
            parts.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        else:
            parts.append(str(payload))
    return "\n".join(parts)

async def _semantic_chain_alignment(
        sample_index: int,
        pred_texts: List[str],  # 预测结果的文本列表
        gt_texts: List[str],    # GT 结果的文本列表
        pred_sqls: List[str],   # 预测 SQL 源码
        gt_sqls: List[str],     # GT SQL 源码
        question: str
    ) -> Tuple[Dict[str, Any], float]:
        

async def _call_output_semantic_judge_async(
    reference_output: str, 
    prediction_output: str, 
    question: str,   
    pred_sql: str = "",
    gt_sql: str = ""
) -> Tuple[Optional[bool], Optional[str]]:
   

def _record_failure(
    extra_infos: List[Dict[str, Any]],
    sample_index: int,
    message: str,
    pred_sqls: List[str],
    gt_sqls: List[str],
    *,
    pred_hashes: Optional[List[str]] = None,
    pred_stmt_hashes: Optional[List[str]] = None,
    gt_hashes: Optional[List[str]] = None,
    pred_db_hash: Optional[str] = None,
    gt_db_hash: Optional[str] = None,
    sqlexe_called: Optional[bool] = None,
) -> None:
    entry = {
        'sample_index': sample_index,
        'mode': 'db_hash_compare',
        'error': message,
        'pred_sql_count': len(pred_sqls),
        'gt_sql_count': len(gt_sqls),
        'pred_result_hashes': pred_hashes or [],
        'pred_stmt_hashes': pred_stmt_hashes or [],
        'gt_result_hashes': gt_hashes or [],
        'pred_db_hash': pred_db_hash,
        'gt_db_hash': gt_db_hash,
        'db_hash_equal': False,
        'soft_result_match': False,
    }
    if sqlexe_called is not None:
        entry['sqlexe_tool_called'] = sqlexe_called
    extra_infos.append(entry)

def _compare_hashes(pred_hash: Optional[str], gt_hash: Optional[str]) -> float:
    if not pred_hash or not gt_hash:
        return 0.0
    return 1.0 if pred_hash == gt_hash else 0.0


def _file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _execute_sequence_and_hash(
    db_file: str,
    sqls: List[str],
    action_statement_counts: Optional[List[int]] = None,
    timeout: int = 5,
    max_rows: int = 100,
    verify_writes: bool = True,
    verify_rows_limit: int = 100,
    is_readonly=False
) -> Tuple[Optional[str], List[str], List[str], List[Optional[Dict[str, Any]]], Optional[str]]:
    tmp_path, conn, err = _open_temp_conn(db_file, timeout=timeout)
    
    



def sql_execution_reward(
    samples: List[Dict[str, Any]],
    return_dict: bool = True,
    db_path: Optional[str] = None,
    max_rows: int = 500,
    timeout: int = 5,
    expand_select_star: bool = True,
    verify_writes: bool = True,
    verify_rows_limit: int = 100,
    # Chain verification options: compare predicted SQL calls against GT actions sequentially
    chain_case_insensitive: bool = False,
    repetition_penalty: float = 0.02,
) -> Any:
    """Score samples by comparing final database hashes after executing SQL sequences."""

    def _collect_gt_sqls(sample_entry: Dict[str, Any]) -> List[str]:
        sqls: List[str] = []
        actions = sample_entry.get('gt_actions')
        if isinstance(actions, (list, np.ndarray)):
            for act in actions:
                name = _safe_get(act, 'name')
                if name not in _VALID_ACTION_NAMES:
                    continue
                args = _safe_get(act, 'arguments') or {}
                sql_text = _safe_get(args, 'sql')
                if isinstance(sql_text, str) and sql_text.strip():
                    sqls.append(sql_text.strip())
        if not sqls:
            gt_sql = sample_entry.get('gt_sql')
            if isinstance(gt_sql, str) and gt_sql.strip():
                sqls.append(gt_sql.strip())
        return sqls

    def _collect_pred_sqls(sample_entry: Dict[str, Any]) -> List[str]:
        sqls: List[str] = []
        raw_solution = sample_entry.get('raw_solution')
        if isinstance(raw_solution, str):
            sqls.extend(_extract_all_sql_from_dialogue(raw_solution))

        pred_sql = sample_entry.get('pred_sql')
        if isinstance(pred_sql, str) and pred_sql.strip():
            extracted = _extract_all_sql_from_dialogue(pred_sql)
            if extracted:
                sqls.extend(extracted)
            else:
                candidate = pred_sql.strip()
                lowered = candidate.lower()
                if lowered.startswith(('select', 'update', 'delete', 'insert', 'with', 'create')) and '<' not in candidate:
                    sqls.append(candidate)

        seen: set[str] = set()
        filtered: List[str] = []
        for item in sqls:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if not cleaned:
                continue
            if cleaned.startswith('<'):
                continue
            # if cleaned not in seen:
            #     seen.add(cleaned)
            filtered.append(cleaned)
        return filtered

    def _gt_action_statement_counts(gt_sqls: List[str]) -> List[int]:
        counts: List[int] = []
        for sql in gt_sqls:
            if not isinstance(sql, str):
                counts.append(1)
                continue
            statements = _split_sql_statements(sql) or []
            valid = [stmt.strip() for stmt in statements if isinstance(stmt, str) and stmt.strip()]
            counts.append(len(valid) if valid else 1)
        return counts
    
    # v3
    async def _score_readonly_semantics_async(
        sample_index: int,
        pred_sqls: List[str],
        sample_db_path: str,
        gt_payloads: List[Dict[str, Any]], 
        sqlexe_invoked: bool,
        gt_sqls_list: List[str],
        question: str = "",
    ) -> Tuple[float, Dict[str, Any]]:
      
    
    
    async def _semantic_chain_alignment(
        sample_index: int,
        pred_texts: List[str],
        gt_texts: List[str],
        pred_sqls: List[str],
        gt_sqls: List[str],
        question: str
    ) -> Tuple[Dict[str, Any], float]:
        alignments = []
        matched = 0
        gt_idx = 0

        

    def _chain_alignment(
        sample_index: int,
        pred_hashes: List[str],
        gt_hashes: List[str],
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        

    async def _process_sample_async(idx: int, sample: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:      


    
    async def _run_all_samples():
        tasks = [_process_sample_async(idx, sample) for idx, sample in enumerate(samples)]
        return await asyncio.gather(*tasks)

    import concurrent.futures
    
    def _run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(_run_all_samples())
        finally:
            new_loop.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        try:
            results = executor.submit(_run_in_new_loop).result()
        except Exception as e:
            _logger.error(f"Batch evaluation failed: {e}")
            results = [(0.0, {'error': str(e)}) for _ in samples]

    rewards = []
    extra_infos = []
    for r, i in results:
        rewards.append(r)
        extra_infos.append(i)

    if torch is not None:
        tensor = torch.tensor(rewards, dtype=torch.bfloat16)
    else:
        tensor = rewards

    if np is not None and not isinstance(tensor, torch.Tensor):
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor, dtype=np.float32)

    extra_info_dict = {'per_sample': extra_infos}
    # Extract specific metrics to top-level lists for validation aggregation
    if extra_infos:
        keys_to_extract = ['db_hash_equal', 'result_hash_equal', 'sqlexe_tool_called']
        for key in keys_to_extract:
            # Use 0.0 (False) as default if key is missing, though it should be present
            values = [float(info.get(key, 0.0)) if info.get(key) is not None else 0.0 for info in extra_infos]
            extra_info_dict[key] = values

    if return_dict:
        return {'reward_tensor': tensor, 'reward_extra_info': extra_info_dict}
    else:
        return tensor


if __name__ == '__main__':
    # quick local test example (won't run here without a db_path and sample SQL)
    example_samples = [
        {'pred_sql': 'SELECT 1 as a', 'gt_sql': 'SELECT 1 as a'},
        {'pred_sql': 'SELECT 1 as a UNION SELECT 2', 'gt_sql': 'SELECT 2 UNION SELECT 1'},
    ]
    print(sql_execution_reward(example_samples, return_dict=True, db_path='/tmp/some.db'))


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        return value.tolist()
    return [value]


def _convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_numpy_types(obj.tolist())
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def sql_compute_score(data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None, **kwargs):
   

def _safe_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return default
