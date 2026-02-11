import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import sys
_this_dir = os.path.dirname(__file__)
_sqlbench_pkg_parent = _this_dir
if _sqlbench_pkg_parent not in sys.path:
    sys.path.insert(0, _sqlbench_pkg_parent)

from MUA_environments.manager import environment_manager
import torch
import numpy as np
from openai import OpenAI

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_PURE_SELECT_MAX_ROWS_DEFAULT = 100
_VALID_ACTION_NAMES = {'sqlexe'}
_SEMANTIC_MODEL_NAME = 'slz-verl-qwen3-32b'
_SEMANTIC_BASE_URL = 'http://slz-nohallu-z32b.bcloud.hb1a-h20.ml.baichuan-inc.com/v1'#'http://172.21.72.116/v1/'

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


def _call_output_semantic_judge(reference_output: str, prediction_output: str) -> Tuple[Optional[bool], Optional[str]]:
    base_url = os.getenv('BASE_URL', _SEMANTIC_BASE_URL)
    api_key = os.getenv('API_KEY', 'EMPTY')
    model = os.getenv('CHAT_MODEL', _SEMANTIC_MODEL_NAME)
    client = OpenAI(base_url=base_url, api_key=api_key)

    system_prompt = (
        'You are a evaluator comparing SQL query outputs between reference ground truth and the predicted output. The reference output determine the result expected by user withs the correct means.'
        'Determine whether the predicted output matches '
        'the reference output semantically. Respond with YES if they represent the same factual content, otherwise NO.(No other options)'
    )
    user_prompt = json.dumps(
        {
            'reference_output': reference_output,
            'prediction_output': prediction_output,
        },
        ensure_ascii=False,
    )
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': 0.0,
    }
    try:
        print(f"DEBUG: [Reward] Calling model {model} at {base_url}. Payload: {payload}")
        response = client.chat.completions.create(**payload)
        print(f"DEBUG: [Reward] Received response: {response}")
        message = response.choices[0].message
        # 只读取 content，不读取 reasoning_content
        content = getattr(message, 'content', '')
        if content is None:
            content = ''
        verdict = content.strip().upper()
        if 'YES' in verdict:
            return True, None
        if 'NO' in verdict:
            return False, None
        return None, f'unexpected_response:{content!r}'
    except Exception as exc:  # pragma: no cover
        _logger.warning('Semantic output judge failed: %s', exc)
        return None, str(exc)


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


def sql_execution_reward(
    samples: List[Dict[str, Any]],
    return_dict: bool = True,
    db_path: Optional[str] = None,
    max_rows: int = 100,
    timeout: int = 5,
    expand_select_star: bool = True,
    verify_writes: bool = True,
    verify_rows_limit: int = 100,
    # Chain verification options: compare predicted SQL calls against GT actions sequentially
    chain_case_insensitive: bool = False,
    repetition_penalty: float = 0.3,
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
    ) -> Tuple[Optional[str], List[str], List[str], List[Optional[Dict[str, Any]]], Optional[str]]:
        tmp_path, conn, err = _open_temp_conn(db_file, timeout=timeout)
        # if err:
        #     if conn:
        #         conn.close()
        #     if tmp_path:
        #         try:
        #             os.remove(tmp_path)
        #         except OSError:
        #             pass
        #     return None, [], [], err
        try:
            stmt_hashes: List[str] = []  # DB hash after each individual statement
            action_hashes: List[str] = []
            stmt_payloads: List[Optional[Dict[str, Any]]] = []
            for sql in sqls:
                statement = sql.strip()
                if not statement:
                    continue
                result = _exec_and_hash_on_conn(
                    conn,
                    statement,
                    max_rows=max_rows,
                    verify_writes=verify_writes,
                    verify_rows_limit=verify_rows_limit,
                )
                payload = result.get('payload') if isinstance(result, dict) else None
                if isinstance(result, dict) and result.get('error'):
                    action_hashes = stmt_hashes.copy()
                    return None, stmt_hashes, action_hashes, stmt_payloads, str(result.get('error'))
                stmt_payloads.append(payload)
                conn.commit()
                current_db_hash = _file_sha256(tmp_path)
                stmt_hashes.append(current_db_hash)
            conn.commit()
            final_db_hash = stmt_hashes[-1] if stmt_hashes else _file_sha256(tmp_path)
            action_hashes = stmt_hashes.copy()
            return final_db_hash, stmt_hashes, action_hashes, stmt_payloads, None
        except Exception as exc:  # pylint: disable=broad-except
            return None, [], [], [], str(exc)
        finally:
            if conn:
                conn.close()
            if tmp_path:
                # try:
                os.remove(tmp_path)
                # except OSError:
                #     pass

    # def _count_matching(pred_sqls: List[str], gt_sqls: List[str], case_insensitive: bool = False) -> int:
    #     gt_counter: Dict[str, int] = {}
    #     for sql in gt_sqls:
    #         key = _normalize_sql_for_compare(sql)
    #         if case_insensitive:
    #             key = key.lower()
    #         gt_counter[key] = gt_counter.get(key, 0) + 1
    #     matches = 0
    #     for sql in pred_sqls:
    #         key = _normalize_sql_for_compare(sql)
    #         if case_insensitive:
    #             key = key.lower()
    #         remaining = gt_counter.get(key, 0)
    #         if remaining > 0:
    #             matches += 1
    #             gt_counter[key] = remaining - 1
    #     return matches

    def _score_readonly_semantics(
        sample_index: int,
        pred_sqls: List[str],
        sample_db_path: str,
        reference_text: str,
        max_rows_hint: int,
        sqlexe_invoked: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        if not pred_sqls:
            penalty = -0.5 if sqlexe_invoked else 0.0
            return penalty, {
                'sample_index': sample_index,
                'mode': 'expsem_semantic',
                'error': 'no_pred_sql_for_readonly',
                'sqlexe_tool_called': sqlexe_invoked,
            }

        pred_payloads: List[Optional[Dict[str, Any]]] = []
        for idx, sql_text in enumerate(pred_sqls):
            result = _hash_sql_result(
                sample_db_path,
                sql_text,
                max_rows=max_rows_hint,
                verify_writes=False,
            )
            if not isinstance(result, dict) or result.get('error'):
                return -0.5, {
                    'sample_index': sample_index,
                    'mode': 'expsem_semantic',
                    'error': f"pred_execution_failed_{idx}",
                    'pred_sql_count': len(pred_sqls),
                }
            payload = _normalize_output_payload(result.get('payload'))
            pred_payloads.append(payload)

        prediction_summary = _payloads_to_text([p for p in pred_payloads if p is not None])
        if not prediction_summary:
            return -0.5, {
                'sample_index': sample_index,
                'mode': 'expsem_semantic',
                'error': 'no_pred_rows_output',
            }

        verdict, err = _call_output_semantic_judge(reference_text, prediction_summary)
        if verdict is None:
            return -0.5, {
                'sample_index': sample_index,
                'mode': 'expsem_semantic',
                'error': err or 'semantic_output_judge_failed',
                'semantic_reference_available': True,
            }
        info = {
            'sample_index': sample_index,
            'mode': 'expsem_semantic',
            'pred_sql_count': len(pred_sqls),
            'gt_sql_count': len(gt_sqls),
            'semantic_match': verdict,
            'semantic_error': err,
            'compare_mode': 'expsem_reference',
        }
        return (1.0 if verdict else 0.0), info

    def _chain_alignment(
        sample_index: int,
        pred_hashes: List[str],
        gt_hashes: List[str],
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        if not gt_hashes:
            return {
                'sample_index': sample_index,
                'mode': 'chain',
                'matched_steps': 0,
                'total_steps': 0,
                'alignments': [],
                'notes': ['empty_ground_truth_hashes'],
            }, 1.0

        alignments: List[Dict[str, Any]] = []
        matched = 0
        gt_index = 0

        for pred_index, pred_hash in enumerate(pred_hashes):
            if gt_index >= len(gt_hashes):
                break
            current_gt_hash = gt_hashes[gt_index]
            entry: Dict[str, Any] = {
                'pred_index': pred_index,
                'gt_index': gt_index,
                'pred_hash': pred_hash,
                'gt_hash': current_gt_hash,
            }
            if pred_hash and current_gt_hash and pred_hash == current_gt_hash:
                entry['match'] = True
                matched += 1
                gt_index += 1
            else:
                entry['match'] = False
            alignments.append(entry)

        total_steps = len(gt_hashes)
        ratio = matched / total_steps if total_steps else 1.0
        info = {
            'sample_index': sample_index,
            'mode': 'chain',
            'matched_steps': matched,
            'total_steps': total_steps,
            'alignments': alignments,
            'chain_ratio': ratio,
        }
        if gt_index < total_steps:
            info['remaining_gt'] = total_steps - gt_index
        return info, ratio

    rewards: List[float] = []
    extra_infos: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        sample_db_path = sample.get('db_path') or db_path
        
        # DEBUG: Check db path and tables
        if sample_db_path and os.path.exists(sample_db_path):
            try:
                import sqlite3
                _conn = sqlite3.connect(sample_db_path)
                _cur = _conn.cursor()
                _cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                _tables = _cur.fetchall()
                _conn.close()
                print(f"[DEBUG] sample_db_path: {sample_db_path}, tables: {_tables}")
            except Exception as e:
                print(f"[DEBUG] sample_db_path: {sample_db_path}, error reading tables: {e}")
        else:
            print(f"[DEBUG] sample_db_path: {sample_db_path} does not exist or is None")

        if not sample_db_path or not os.path.exists(sample_db_path):
            _record_failure(extra_infos, idx, 'missing_db_path', [], [])
            rewards.append(0.0)
            continue

        gt_sqls = _collect_gt_sqls(sample)
        pred_sqls = _collect_pred_sqls(sample)

        # Calculate repetition penalty
        unique_sqls = set(pred_sqls)
        repetition_count = len(pred_sqls) - len(unique_sqls)
        penalty_score = repetition_count * repetition_penalty

        raw_solution_text = sample.get('raw_solution')
        sqlexe_called = _text_has_sqlexe_tool_call(raw_solution_text) or _text_has_sqlexe_tool_call(sample.get('pred_sql'))
        gt_action_counts = _gt_action_statement_counts(gt_sqls) if gt_sqls else []

        reference_text = sample.get('gt_expsem')
        if isinstance(reference_text, str):
            reference_text = reference_text.strip()
        else:
            reference_text = None

        gt_result_hashes: List[str] = sample.get('gt_result_hashes') or []
        gt_final_hash: Optional[str] = None
        if gt_result_hashes:
            gt_final_hash = gt_result_hashes[-1]
        else:
            gt_final_hash = sample.get('gt_data_hash')

        if reference_text:
            reward_value, info = _score_readonly_semantics(
                idx,
                pred_sqls,
                sample_db_path,
                reference_text,
                _PURE_SELECT_MAX_ROWS_DEFAULT,
                sqlexe_called,
            )
            extra_infos.append(info)
            rewards.append(reward_value - penalty_score)
            continue

        if not pred_sqls and gt_sqls:
            message = 'no_pred_sql_extracted_sqlexe_called' if sqlexe_called else 'no_pred_sql_without_sqlexe_call'
            penalty = -0.5 if sqlexe_called else 0.0
            _record_failure(
                extra_infos,
                idx,
                message,
                pred_sqls,
                gt_sqls,
                sqlexe_called=sqlexe_called,
            )
            rewards.append(penalty - penalty_score)
            continue

        action_counts_arg = gt_action_counts if gt_action_counts else None
        pred_hash, pred_stmt_hashes, pred_action_hashes, _, pred_error = _execute_sequence_and_hash(
            sample_db_path,
            pred_sqls,
            action_statement_counts=action_counts_arg,
        )
        if pred_error:
            pred_final_hash = pred_action_hashes[-1] if pred_action_hashes else pred_hash
            _record_failure(
                extra_infos,
                idx,
                f'pred_execution_failed: {pred_error}',
                pred_sqls,
                gt_sqls,
                pred_hashes=pred_action_hashes,
                pred_stmt_hashes=pred_stmt_hashes,
                pred_db_hash=pred_final_hash,
                gt_db_hash=gt_final_hash,
            )
            rewards.append(0.0 - penalty_score)
            continue

        pred_final_hash = pred_action_hashes[-1] if pred_action_hashes else pred_hash
        hashes_equal = bool(pred_final_hash) and bool(gt_final_hash) and pred_final_hash == gt_final_hash
        chain_info, chain_ratio = _chain_alignment(idx, pred_action_hashes, gt_result_hashes)
        if chain_ratio is None:
            chain_ratio = 0.0
        matched_steps = chain_info.get('matched_steps', 0)
        if not hashes_equal:
            reward_value = 0.0
        else:
            # Grant 1 point for matching final DB hash plus additional points per aligned step
            reward_value = 1.0 + float(matched_steps)

        result_hashes_equal = bool(chain_ratio is not None and matched_steps == len(gt_result_hashes))

        extra_infos.append({
            'sample_index': idx,
            'mode': 'db_hash_compare',
            'pred_db_hash': pred_final_hash,
            'pred_db_file_hash': pred_hash,
            'gt_db_hash': gt_final_hash,
            'db_hash_equal': hashes_equal,
            'pred_result_hashes': pred_action_hashes,
            'pred_stmt_hashes': pred_stmt_hashes,
            'gt_result_hashes': gt_result_hashes,
            'result_hash_equal': result_hashes_equal,
            'soft_result_match': False,
            'pred_sql_count': len(pred_sqls),
            'gt_sql_count': len(gt_sqls),
            'chain_alignment': chain_info,
            'chain_ratio': chain_ratio,
            'error': None,
        })
        rewards.append(reward_value - penalty_score)

    if torch is not None:
        tensor = torch.tensor(rewards, dtype=torch.bfloat16)
    else:
        tensor = rewards

    if np is not None and not isinstance(tensor, torch.Tensor):
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor, dtype=np.float32)

    if return_dict:
        return {'reward_tensor': tensor, 'reward_extra_info': {'per_sample': extra_infos}}
    else:
        return tensor


# Small adapter example: if your DataProto is different, write a wrapper that maps it to samples list.
# Example adapter for the Parquet-row style records used in this repo (where reward_model.ground_truth.actions
# contains SQL under actions[].arguments.sql):
# def data_proto_to_samples(data_proto):
#     samples = []
#     for ex in data_proto.examples:  # adapt this to your DataProto
#         pred_sql = ex.get('model_sql') or ex.get('prediction')  # however the LLM output is stored
#         # Extract first sql from ground truth actions
#         gt_actions = ex.get('reward_model', {}).get('ground_truth', {}).get('actions', [])
#         gt_sql = None
#         if gt_actions:
#             gt_sql = gt_actions[0].get('arguments', {}).get('sql')
#         samples.append({'pred_sql': pred_sql, 'gt_sql': gt_sql})
#     return samples


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
    """Adapter bridging Batch/Naive reward manager inputs to ``sql_execution_reward``."""
    taubench_database = kwargs.pop('taubench_database', None)

    # Legacy knob retained for API compatibility; hash map no longer used in fast path.
    kwargs.pop('ignore_gt_hash_map', False)
    expand_select_star: bool = bool(kwargs.pop('expand_select_star', True))
    verify_writes: bool = bool(kwargs.pop('verify_writes', True))
    verify_rows_limit: int = int(kwargs.pop('verify_rows_limit', 1000))
    chain_case_insensitive: bool = bool(kwargs.pop('chain_case_insensitive', False))
    kwargs.pop('messages', None)

    single_call = False
    if data_sources is None and solution_strs is None and ground_truths is None:
        if any(key in kwargs for key in ('data_source', 'solution_str', 'ground_truth')):
            single_call = True
            data_sources = [kwargs.pop('data_source', None)]
            solution_strs = [kwargs.pop('solution_str', kwargs.pop('solution', None))]
            ground_truths = [kwargs.pop('ground_truth', None)]
            extra_infos = [kwargs.pop('extra_info', None)]
        else:
            data_sources = []
            solution_strs = []
            ground_truths = []

    data_sources_list = _ensure_list(data_sources)
    solution_list = _ensure_list(solution_strs)
    ground_truth_list = _ensure_list(ground_truths)
    extra_info_list = _ensure_list(extra_infos)

    # Normalize inputs to ensure Python native types (handling NumPy arrays/scalars from data loaders)
    ground_truth_list = [_convert_numpy_types(gt) for gt in ground_truth_list]
    extra_info_list = [_convert_numpy_types(ei) for ei in extra_info_list]

    taubench_db_list = _ensure_list(taubench_database)
    ability_cache: Dict[str, Optional[str]] = {}

    lengths = [len(solution_list), len(ground_truth_list), len(data_sources_list), len(extra_info_list)]
    n = max(lengths) if any(lengths) else 0
    if single_call and n == 0:
        n = 1
    if n == 0:
        return {'score': 0.0} if single_call else []

    default_db_path = kwargs.get('db_path')

    def _db_path_from_entry(entry: Any) -> Optional[str]:
        if isinstance(entry, dict):
            return entry.get('db_path') or entry.get('path')
        return None

    def _taubench_db_path(index: int) -> Optional[str]:
        if not taubench_db_list:
            return None
        if index < len(taubench_db_list):
            path = _db_path_from_entry(taubench_db_list[index])
            if path:
                return path
        for entry in taubench_db_list:
            path = _db_path_from_entry(entry)
            if path:
                return path
        return None

    samples: List[Dict[str, Any]] = []

    for i in range(n):
        pred_sql = solution_list[i] if i < len(solution_list) else None
        if not isinstance(pred_sql, str):
            pred_sql = _extract_sql_from_obj(pred_sql)

        gt_obj = ground_truth_list[i] if i < len(ground_truth_list) else None
        gt_sql = _extract_sql_from_obj(gt_obj)

        extra = extra_info_list[i] if i < len(extra_info_list) else None

        gt_payload = None
        gt_expsem: Optional[str] = None
        gt_result_hashes: List[str] = []
        gt_data_hash: Optional[str] = None
        if isinstance(gt_obj, dict):
            if 'ground_truth' in gt_obj and isinstance(gt_obj['ground_truth'], dict):
                gt_payload = gt_obj['ground_truth']
            else:
                gt_payload = gt_obj
        if isinstance(gt_payload, dict):
            candidate = gt_payload.get('expsem_for_readonly')
            if isinstance(candidate, str):
                text = candidate.strip()
                if text:
                    gt_expsem = text
            rh_entries = gt_payload.get('result_hashes')
            if isinstance(rh_entries, (list, np.ndarray)):
                for entry in rh_entries:
                    hsh = _safe_get(entry, 'hash')
                    if isinstance(hsh, str) and hsh:
                        gt_result_hashes.append(hsh)
            data_hash = gt_payload.get('gt_data_hash')
            if isinstance(data_hash, str) and data_hash:
                gt_data_hash = data_hash

        sample_db_path = None
        if i < len(data_sources_list):
            ds_entry = data_sources_list[i]
            if isinstance(ds_entry, dict):
                sample_db_path = ds_entry.get('db_path') or ds_entry.get('path')
                if not sample_db_path:
                    env_info = ds_entry.get('environment')
                    if isinstance(env_info, dict):
                        sample_db_path = env_info.get('db_path') or env_info.get('path')
            elif isinstance(ds_entry, str):
                cached = ability_cache.get(ds_entry)
                if cached is None and ds_entry not in ability_cache:
                    shared = environment_manager.get_new_environment_data(ds_entry) or {}
                    cached = _db_path_from_entry(shared)
                    ability_cache[ds_entry] = cached
                sample_db_path = cached
        if sample_db_path is None:
            sample_db_path = _taubench_db_path(i)
        if sample_db_path is None:
            sample_db_path = default_db_path

        samples.append({
            'pred_sql': pred_sql,
            'gt_sql': gt_sql,
            'extra_info': extra,
            'extra': extra,
            'db_path': sample_db_path,
            'raw_solution': solution_list[i] if i < len(solution_list) else None,
            'gt_actions': (gt_payload.get('actions') if isinstance(gt_payload, dict) else None),
            'gt_expsem': gt_expsem,
            'gt_result_hashes': gt_result_hashes,
            'gt_data_hash': gt_data_hash,
            'question': (
                extra.get('question') if isinstance(extra, dict) and extra.get('question')
                else (gt_payload.get('question') if isinstance(gt_payload, dict) and gt_payload.get('question') else None)
            ),
        })

    allowed = {k: kwargs[k] for k in ('db_path', 'max_rows', 'timeout') if k in kwargs}
    result = sql_execution_reward(
        samples,
        return_dict=True,
        expand_select_star=expand_select_star,
        verify_writes=verify_writes,
        verify_rows_limit=verify_rows_limit,
        chain_case_insensitive=chain_case_insensitive,
        **allowed,
    )

    reward_tensor = result.get('reward_tensor')
    reward_extra = result.get('reward_extra_info', {})

    if reward_tensor is None:
        raw_scores = [0.0] * len(samples)
    elif torch is not None and isinstance(reward_tensor, torch.Tensor):
        safe_tensor = reward_tensor.detach()
        if safe_tensor.is_cuda:
            safe_tensor = safe_tensor.cpu()
        if safe_tensor.dtype == torch.bfloat16:
            safe_tensor = safe_tensor.to(torch.float32)
        raw_scores = safe_tensor.numpy().tolist()
    elif hasattr(reward_tensor, 'tolist'):
        raw_scores = reward_tensor.tolist()
    else:
        raw_scores = list(reward_tensor)

    scores: List[float] = [float(value) for value in raw_scores]

    per_sample_infos = reward_extra.get('per_sample', [])
    out: List[Dict[str, Any]] = []
    for i, score in enumerate(scores):
        info: Dict[str, Any] = {'score': score}
        if i < len(per_sample_infos) and isinstance(per_sample_infos[i], dict):
            # Convert NumPy types to Python types for JSON serialization
            converted_info = _convert_numpy_types(per_sample_infos[i])
            info.update(converted_info)
        out.append(info)

    if single_call:
        return out[0] if out else {'score': 0.0}
    return out

def _safe_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return default
