#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verl.utils.reward_score.sqlbench_pkg.db_exec import (
    _open_temp_conn,
    _exec_and_hash_on_conn,
)
from verl.utils.reward_score.sqlbench_pkg.sql_utils import _sql_key
WORKSPACE_ROOT = PROJECT_ROOT.parent
PREFERRED_TOOL_ROOT = PROJECT_ROOT / "verl" / "tools" / "sqlbench_apigen"
PREFERRED_DATA_ROOT = PREFERRED_TOOL_ROOT / "data"
LEGACY_TAUBENCH_ROOT = WORKSPACE_ROOT / "data_pipe" / "RawData" / "TAUBench"
# Helper duplicated from sqlbench reward tooling to keep dataset self-contained.
def _file_sha256(path: str) -> Optional[str]:
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None



def _prefer_existing(primary: Path, fallback: Path) -> Path:
    """Return the first path that exists, defaulting to the primary when both miss."""
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


DEFAULT_AIRLINE_DB_PATH = _prefer_existing(
    PREFERRED_DATA_ROOT / "airline" / "airline.sqlite",
    LEGACY_TAUBENCH_ROOT / "taubench_airline.sqlite",
)
DEFAULT_AIRLINE_SCHEMA_PATH = _prefer_existing(
    PREFERRED_DATA_ROOT / "airline" / "schema.sql",
    LEGACY_TAUBENCH_ROOT / "schema.sql",
)
# Retail paths removed (airline-only conversion).

import pyarrow as pa
import pyarrow.parquet as pq

SQL_AGENT_POLICY = (
    "# SQL Agent Policy\n"
    "You are a multiturn Text-to-SQL Agent.\n"
    "Your task is to understand the user's question step by step and make tool calls to finally generate the correct SQL query to solve it.\n\n"

    "## Tools:\n"
    "### getdbschema:\n"
    "- Call this tool to retrieve the FULL database schema. Do NOT pass any table names as arguments; this tool always returns the complete schema for the entire database.\n\n"
    "### sqlexe:\n"
    "- Call this tool to generate and execute SQL queries against the database and obtain results for verification and further analysis.\n\n"

    "## Rules:\n"
    
    "## Current Task Description\n"
    
    "### Knowledge Base\n"
    
    "#### Retrieval Guidance: "
    "#### Mutation Protocol: "
    "### Safety & Verification:\n"
   )

USER_SIM_PROMPT_TEMPLATE = (
    "You are playing the role of the USER (customer) interacting with a SQL assistant agent. You must act as the customer who needs help.\n\n"
    "Follow these rules to simulate a realistic dialogue: \n"
    "- Only reveal information that the agent explicitly asks for.\n"
    "- Avoid inventing IDs if you do not know them; say you do not have them.\n"
    "- When the goal is achieved, answer with '###STOP###' alone to end.\n"
    "- Paraphrase the initial intent instead of repeating it verbatim.\n\n"
    
    "Instructions: \\n\\n"
    "{instruction}\\n\n"

    "Rules:\\n"


    
    "- CRITICAL: When mentioning specific entities (like airport codes, city names, IDs, dates, flight numbers, etc.) from the instruction, you MUST use the EXACT value provided. Do NOT paraphrase, translate, or expand them (e.g., do NOT change 'PHX' to 'Phoenix', do NOT change '2024-05-19' to 'May 19th'). Copy them exactly as they appear.\n"
    "/no_think"
)

DEFAULT_FIRST_USER = "Hi, I need help with my flight reservation."

MAX_ROWS = 100


def _first_turn_utterance(first_user: str) -> str:
    text = (first_user or "").strip()
    if not text:
        return DEFAULT_FIRST_USER
    first_line = text.splitlines()[0].strip()
    return first_line or DEFAULT_FIRST_USER


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    observation: Any
    human_messages: Sequence[str]


@dataclass
class SqlAction:
    tool_name: str
    sql: str
    description: str
    arguments: Dict[str, Any]
    observation: Any
    tool_call_index: Optional[int] = None
    split_index: Optional[int] = None
    split_total: Optional[int] = None
    source_action_index: Optional[int] = None


@dataclass
class DomainConfig:
    ability: str
    data_source: str
    db_name: str
    db_path: Optional[str]
    schema_path: Optional[str]
    converters: Dict[str, Callable[[ToolCall, Dict[str, Any]], Optional[SqlAction]]]
    fallback_converter: Callable[[ToolCall, Dict[str, Any]], Optional[SqlAction]]


def _split_sql_statements(sql_text: str) -> List[str]:
    if not isinstance(sql_text, str):
        return []
    statements: List[str] = []
    current: List[str] = []
    in_single = False
    in_double = False
    escape = False
    for ch in sql_text:
        if ch == "\\" and not escape:
            escape = True
            current.append(ch)
            continue
        if ch == "'" and not in_double and not escape:
            in_single = not in_single
        elif ch == '"' and not in_single and not escape:
            in_double = not in_double
        if ch == ';' and not in_single and not in_double:
            segment = ''.join(current).strip()
            if segment:
                statements.append(segment)
            current = []
        else:
            current.append(ch)
        escape = False
    tail = ''.join(current).strip()
    if tail:
        statements.append(tail)
    return statements or [sql_text.strip()]


def expand_sql_actions(actions: Sequence[SqlAction]) -> List[SqlAction]:
    expanded: List[SqlAction] = []
    for action in actions:
        sql_text = action.sql or ""
        statements = _split_sql_statements(sql_text)
        if len(statements) <= 1:
            if action.split_total is None:
                action.split_index = 1
                action.split_total = 1
            expanded.append(action)
            continue
        total = len(statements)
        for idx, stmt in enumerate(statements, start=1):
            trimmed = stmt.rstrip() + (";" if not stmt.rstrip().endswith(';') else "")
            new_action = SqlAction(
                tool_name=action.tool_name,
                sql=trimmed,
                description=f"{action.description} [part {idx}/{total}]",
                arguments=action.arguments,
                observation=action.observation,
                tool_call_index=action.tool_call_index,
                split_index=idx,
                split_total=total,
                source_action_index=action.source_action_index,
            )
            expanded.append(new_action)
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TauBench tool-use to SQL converter")
    parser.add_argument("--input", required=True, help="Path to apigen-mt_*.json file")
    parser.add_argument("--output", required=True, help="Destination parquet path")
    parser.add_argument("--ability", default=None, help="Ability label for airline domain (legacy flag)")
    parser.add_argument("--data-source", dest="data_source", default=None, help="Data source tag for airline domain")
    parser.add_argument("--db-name", dest="db_name", default=None, help="Airline database name exposed to tools")
    parser.add_argument("--db-path",dest="db_path",default=None,help="Override airline sqlite path (defaults to TauBench airline DB)")
    parser.add_argument("--schema-path",dest="schema_path",default=None,help="Override airline schema path (defaults to TauBench airline schema.sql)")
    parser.add_argument("--min-actions", type=int, default=1, help="Skip dialogs with fewer SQL actions")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of dialogs processed")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logger level")
    return parser.parse_args()


def setup_logger(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="[%(levelname)s] %(message)s")


def safe_json_loads(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def normalise_path(path: Optional[str]) -> Optional[str]:
    """Return workspace-relative string for the provided path when possible."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (WORKSPACE_ROOT / p).resolve()
    else:
        p = p.resolve()
    try:
        return str(p.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(p)


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return "NULL"
    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def join_sql(lines: Iterable[str]) -> str:
    return "\n".join(line for line in lines if line)


def is_read_query(sql: str) -> bool:
    text = (sql or "").lstrip().lower()
    if not text:
        return False
    if text.startswith("select"):
        return True
    if text.startswith("with"):
        lowered = " ".join(text.split())
        write_markers = (" insert ", " update ", " delete ", " merge ", " replace ", " create ", " drop ", " alter ", "pragma ")
        return not any(marker in lowered for marker in write_markers)
    return False


def summarise_passengers(passengers: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for passenger in passengers:
        first = passenger.get("first_name", "?")
        last = passenger.get("last_name", "?")
        dob = passenger.get("dob") or passenger.get("date_of_birth")
        if dob:
            parts.append(f"{first} {last} ({dob})")
        else:
            parts.append(f"{first} {last}")
    return ", ".join(parts)


def extract_observed_created_at(call: ToolCall) -> Optional[str]:
    """Return the created_at value from the tool observation if present."""
    obs = call.observation
    if isinstance(obs, str):
        obs = safe_json_loads(obs)
    if isinstance(obs, dict):
        value = obs.get("created_at")
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed
    return None

def describe_flights(flights: Sequence[Dict[str, Any]]) -> str:
    return ", ".join(f"{seg.get('flight_number', '?')} on {seg.get('date', '?')}" for seg in flights)


def convert_airline_get_user_details(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    user_id = call.arguments.get("user_id")
    if not user_id:
        return None
    sql = "SELECT * FROM users WHERE user_id = {uid};".format(uid=sql_literal(user_id))
    description = f"Retrieve full profile (identity, membership, dob, address) for user_id={user_id}."
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def convert_airline_get_reservation_details(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    reservation_id = call.arguments.get("reservation_id")
    if not reservation_id:
        return None
    obs_user = None
    if isinstance(call.observation, dict):
        obs_user = call.observation.get("user_id")
    sql = "SELECT * FROM reservations WHERE reservation_id = {rid};".format(
        rid=sql_literal(reservation_id)
    )
    if obs_user:
        description = f"Look up reservation {reservation_id} for user_id={obs_user}."
    else:
        description = f"Look up reservation {reservation_id} to authenticate the user."
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def convert_airline_list_all_airports(call: ToolCall, meta: Dict[str, Any]) -> SqlAction:
    sql = "SELECT code, city, state FROM airports ORDER BY code;"
    description = "Enumerate every airport code with its city and state."
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def convert_airline_search_direct_flight(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    origin = call.arguments.get("origin")
    destination = call.arguments.get("destination")
    date = call.arguments.get("date")
    if not (origin and destination and date):
        return None
    sql = (
        "SELECT "
        "    flight_number, "
        "    origin, "
        "    destination, "
        "    travel_date AS date, "
        "    scheduled_departure_time_est, "
        "    scheduled_arrival_time_est, "
        "    price_basic_economy, "
        "    price_economy, "
        "    price_business, "
        "    available_basic_economy, "
        "    available_economy, "
        "    available_business, "
        "    status "
        "FROM airline_direct_flight_listing "
        "WHERE origin = {o} "
        "  AND destination = {d} "
        "  AND travel_date = {dt} "
        "ORDER BY scheduled_departure_time_est;"
    ).format(o=sql_literal(origin), d=sql_literal(destination), dt=sql_literal(date))
    description = (
        f"List available direct flights on {date} from {origin} to {destination} with schedule, availability, fares, and status."
    )
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def convert_airline_search_onestop_flight(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    origin = call.arguments.get("origin")
    destination = call.arguments.get("destination")
    date = call.arguments.get("date")
    if not (origin and destination and date):
        return None
    sql = (
        "SELECT "
        "    origin, "
        "    destination, "
        "    layover_airport, "
        "    first_travel_date AS first_leg_date, "
        "    first_flight_number, "
        "    first_departure_time_est, "
        "    first_arrival_time_est, "
        "    first_status, "
        "    first_available_basic_economy, "
        "    first_available_economy, "
        "    first_available_business, "
        "    first_price_basic_economy, "
        "    first_price_economy, "
        "    first_price_business, "
        "    second_travel_date AS second_leg_date, "
        "    second_flight_number, "
        "    second_departure_time_est, "
        "    second_arrival_time_est, "
        "    second_status, "
        "    second_available_basic_economy, "
        "    second_available_economy, "
        "    second_available_business, "
        "    second_price_basic_economy, "
        "    second_price_economy, "
        "    second_price_business "
        "FROM airline_onestop_itinerary_listing "
        "WHERE origin = {o} "
        "  AND destination = {d} "
        "  AND first_travel_date = {dt} "
        "ORDER BY first_departure_time_est, second_departure_time_est;"
    ).format(o=sql_literal(origin), d=sql_literal(destination), dt=sql_literal(date))
    description = (
        f"List one-stop itineraries on {date} from {origin} to {destination} with both legs' schedules, availability, fares, and connection airport."
    )
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def convert_airline_calculate(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    expression = call.arguments.get("expression")
    if not expression:
        return None
    cleaned = expression.replace(";", "")
    sql = f"SELECT {cleaned} AS result;"
    description = f"Evaluate expression '{cleaned}' to support fee calculation."
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def convert_airline_cancel_reservation(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    reservation_id = call.arguments.get("reservation_id")
    if not reservation_id:
        return None
    # Schema has no cancelled_at column; policy specifies only status change.
    # However, original tool appends negative payments (refunds) to history.
    rid = sql_literal(reservation_id)
    sql = (
        "INSERT INTO reservation_payments (reservation_id, payment_index, payment_id, amount) "
        "SELECT reservation_id, "
        "(SELECT COALESCE(MAX(payment_index), -1) FROM reservation_payments WHERE reservation_id = {rid}) + "
        "ROW_NUMBER() OVER (ORDER BY payment_index), "
        "payment_id, -amount "
        "FROM reservation_payments "
        "WHERE reservation_id = {rid};\n"
        "UPDATE reservations SET status = 'cancelled' "
        "WHERE reservation_id = {rid};"
    ).format(rid=rid)
    description = f"Cancel reservation {reservation_id} (refund payments and set status='cancelled')."
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)

# 有简化
def convert_airline_update_reservation_baggages(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    reservation_id = call.arguments.get("reservation_id")
    payment_id = call.arguments.get("payment_id")
    if not reservation_id:
        return None
    total = to_int(call.arguments.get("total_baggages"))
    nonfree = to_int(call.arguments.get("nonfree_baggages"))
    
    updates = []
    if total is not None:
        updates.append(f"total_baggages = {total}")
    if nonfree is not None:
        updates.append(f"nonfree_baggages = {nonfree}")
        
    if not updates:
        return None

    rid = sql_literal(reservation_id)
    pid = sql_literal(payment_id)
    
    sql_lines = []
    
    # We must calculate the price difference BEFORE updating the reservation,
    # because the update will overwrite the old 'nonfree_baggages' value needed for the calculation.
    if nonfree is not None and payment_id:
        # 1. Insert payment record (only if nonfree increases)
        sql_lines.append(
            "INSERT INTO reservation_payments (reservation_id, payment_index, payment_id, amount) "
            "SELECT {rid}, (SELECT COALESCE(MAX(payment_index), -1) + 1 FROM reservation_payments WHERE reservation_id = {rid}), {pid}, "
            "({nonfree} - r.nonfree_baggages) * 50 "
            "FROM reservations r WHERE r.reservation_id = {rid} AND {nonfree} > r.nonfree_baggages;".format(
                rid=rid, pid=pid, nonfree=nonfree
            )
        )

    # 2. Update reservation baggages
    sql_lines.append(
        f"UPDATE reservations SET {', '.join(updates)} WHERE reservation_id = {rid};"
    )
    
    sql = join_sql(sql_lines)
    
    description = (
        f"Adjust baggage counts for reservation {reservation_id}: {', '.join(updates)}. "
        f"Charge difference to {payment_id}."
    )
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def convert_airline_update_reservation_flights(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    reservation_id = call.arguments.get("reservation_id")
    flights = call.arguments.get("flights") or []
    cabin = call.arguments.get("cabin")
    if not (reservation_id and flights and cabin):
        return None
    sql_lines = [
        #"BEGIN TRANSACTION;",
        f"UPDATE reservations SET cabin = {sql_literal(cabin)} WHERE reservation_id = {sql_literal(reservation_id)};",
        f"DELETE FROM reservation_flights WHERE reservation_id = {sql_literal(reservation_id)};",
    ]
    for idx, seg in enumerate(flights):
        flight_no = seg.get("flight_number")
        flight_date = seg.get("date")
        # Use INSERT INTO ... SELECT to populate origin, destination, price from schedules
        sql_lines.append(
            "INSERT INTO reservation_flights (reservation_id, segment_index, flight_number, flight_date, origin, destination, price) "
            "SELECT {rid}, {idx}, {fno}, {fdate}, f.origin, f.destination, "
            "CASE {cab} "
            "WHEN 'basic_economy' THEN s.price_basic_economy "
            "WHEN 'economy' THEN s.price_economy "
            "WHEN 'business' THEN s.price_business "
            "ELSE s.price_economy END "
            "FROM flights f JOIN flight_schedules s ON f.flight_number = s.flight_number "
            "WHERE f.flight_number = {fno} AND s.departure_date = {fdate};".format(
                rid=sql_literal(reservation_id),
                idx=idx,
                fno=sql_literal(flight_no),
                fdate=sql_literal(flight_date),
                cab=sql_literal(cabin)
            )
        )
    # sql_lines.append("COMMIT;")
    sql = join_sql(sql_lines)
    description = (
        f"Replace flights for reservation {reservation_id} with segments [{describe_flights(flights)}] in {cabin} cabin."
    )
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def convert_airline_update_reservation_passengers(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    reservation_id = call.arguments.get("reservation_id")
    passengers = call.arguments.get("passengers") or []
    if not (reservation_id and passengers):
        return None
    sql_lines = [
        #"BEGIN TRANSACTION;",
        f"DELETE FROM reservation_passengers WHERE reservation_id = {sql_literal(reservation_id)};",
    ]
    for idx, passenger in enumerate(passengers):
        sql_lines.append(
            "INSERT INTO reservation_passengers (reservation_id, passenger_index, first_name, last_name, dob) "
            "VALUES ({rid}, {idx}, {first}, {last}, {dob});".format(
                rid=sql_literal(reservation_id),
                idx=idx,
                first=sql_literal(passenger.get("first_name")),
                last=sql_literal(passenger.get("last_name")),
                dob=sql_literal(passenger.get("dob") or passenger.get("date_of_birth")),
            )
        )
    # sql_lines.append("COMMIT;")
    sql = join_sql(sql_lines)
    description = (
        f"Overwrite passengers for reservation {reservation_id} with [{summarise_passengers(passengers)}]."
    )
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def convert_airline_send_certificate(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    user_id = call.arguments.get("user_id")
    amount = to_float(call.arguments.get("amount"))
    if not user_id or amount is None:
        return None
    sql = (
        "WITH candidates(payment_id) AS (\n"
        "    VALUES ('certificate_3221322'),\n"
        "           ('certificate_3221323'),\n"
        "           ('certificate_3221324')\n"
        "), chosen AS (\n"
        "    SELECT payment_id\n"
        "    FROM candidates\n"
        "    WHERE NOT EXISTS (\n"
        "        SELECT 1\n"
        "        FROM user_payment_methods upm\n"
        "        WHERE upm.user_id = {user}\n"
        "          AND upm.payment_id = candidates.payment_id\n"
        "    )\n"
        "    LIMIT 1\n"
        ")\n"
        "INSERT INTO user_payment_methods (user_id, payment_id, source, brand, last_four, amount)\n"
        "SELECT {user}, payment_id, 'certificate', NULL, NULL, {amount}\n"
        "FROM chosen;"
    ).format(user=sql_literal(user_id), amount=sql_literal(amount))
    description = (
        f"Add a certificate payment method for user_id={user_id} (amount {amount:.2f}) using the first available designated ID."
    )
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


def generate_reservation_id(call: ToolCall, meta: Dict[str, Any]) -> str:
    if isinstance(call.observation, dict):
        obs_res = call.observation.get("reservation_id")
        if obs_res:
            return str(obs_res)
    seed = json.dumps(
        {
            "conversation": meta["conversation_index"],
            "action": meta["action_index"],
            "user_id": call.arguments.get("user_id"),
            "origin": call.arguments.get("origin"),
            "destination": call.arguments.get("destination"),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest().upper()
    return f"GEN{digest[:10]}"


def convert_airline_book_reservation(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    user_id = call.arguments.get("user_id")
    origin = call.arguments.get("origin")
    destination = call.arguments.get("destination")
    flight_type = call.arguments.get("flight_type")
    cabin = call.arguments.get("cabin")
    flights = call.arguments.get("flights") or []
    passengers = call.arguments.get("passengers") or []
    payment_methods = call.arguments.get("payment_methods") or []
    total_baggages = to_int(call.arguments.get("total_baggages"))
    nonfree_baggages = to_int(call.arguments.get("nonfree_baggages"))
    insurance = call.arguments.get("insurance")
    if not (user_id and origin and destination and flight_type and cabin and flights and passengers):
        return None
    reservation_id = generate_reservation_id(call, meta)
    sql_lines = [
        #"BEGIN TRANSACTION;",
        (
            "INSERT OR REPLACE INTO reservations (reservation_id, user_id, origin, destination, flight_type, cabin, "
            "total_baggages, nonfree_baggages, insurance, status) "
            "VALUES ({rid}, {user}, {origin}, {dest}, {ftype}, {cabin}, {total}, {nonfree}, {ins}, 'confirmed');"
        ).format(
            rid=sql_literal(reservation_id),
            user=sql_literal(user_id),
            origin=sql_literal(origin),
            dest=sql_literal(destination),
            ftype=sql_literal(flight_type),
            cabin=sql_literal(cabin),
            total=total_baggages if total_baggages is not None else "NULL",
            nonfree=nonfree_baggages if nonfree_baggages is not None else "NULL",
            ins=sql_literal(insurance),
        ),
    ]
    for idx, seg in enumerate(flights):
        flight_no = seg.get("flight_number")
        flight_date = seg.get("date")
        sql_lines.append(
            "INSERT INTO reservation_flights (reservation_id, segment_index, flight_number, flight_date, origin, destination, price) "
            "SELECT {rid}, {idx}, {fno}, {fdate}, f.origin, f.destination, "
            "CASE {cab} "
            "WHEN 'basic_economy' THEN s.price_basic_economy "
            "WHEN 'economy' THEN s.price_economy "
            "WHEN 'business' THEN s.price_business "
            "ELSE s.price_economy END "
            "FROM flights f JOIN flight_schedules s ON f.flight_number = s.flight_number "
            "WHERE f.flight_number = {fno} AND s.departure_date = {fdate};".format(
                rid=sql_literal(reservation_id),
                idx=idx,
                fno=sql_literal(flight_no),
                fdate=sql_literal(flight_date),
                cab=sql_literal(cabin)
            )
        )
    for idx, passenger in enumerate(passengers):
        sql_lines.append(
            "INSERT INTO reservation_passengers (reservation_id, passenger_index, first_name, last_name, dob) "
            "VALUES ({rid}, {idx}, {first}, {last}, {dob});".format(
                rid=sql_literal(reservation_id),
                idx=idx,
                first=sql_literal(passenger.get("first_name")),
                last=sql_literal(passenger.get("last_name")),
                dob=sql_literal(passenger.get("dob") or passenger.get("date_of_birth")),
            )
        )
    for pidx, payment in enumerate(payment_methods):
        if isinstance(payment, dict):
            pid = payment.get("payment_id")
            amount = to_float(payment.get("amount"))
        else:
            pid = str(payment)
            amount = None
        sql_lines.append(
            "INSERT INTO reservation_payments (reservation_id, payment_index, payment_id, amount) "
            f"VALUES ({sql_literal(reservation_id)}, {pidx}, {sql_literal(pid)}, {amount if amount is not None else 'NULL'});"
        )
    # sql_lines.append("COMMIT;")
    sql = join_sql(sql_lines)
    description = (
        f"Create reservation {reservation_id} for user_id={user_id} from {origin} to {destination} ({flight_type}) in {cabin} cabin, "
        f"covering {len(flights)} segments, {len(passengers)} passengers, total_baggages={total_baggages if total_baggages is not None else 'NULL'}, "
        f"nonfree_baggages={nonfree_baggages if nonfree_baggages is not None else 'NULL'}, insurance={insurance}."
    )
    return SqlAction(call.tool_name, sql, description, call.arguments, call.observation)


    # Retail conversion functions removed.


def convert_airline_fallback(call: ToolCall, meta: Dict[str, Any]) -> Optional[SqlAction]:
    logging.debug("Skip unsupported airline tool '%s'", call.tool_name)
    return None


AIRLINE_TOOL_CONVERTERS = {
    "get_user_details": convert_airline_get_user_details,
    "get_reservation_details": convert_airline_get_reservation_details,
    "list_all_airports": convert_airline_list_all_airports,
    "search_direct_flight": convert_airline_search_direct_flight,
    "search_onestop_flight": convert_airline_search_onestop_flight,
    "calculate": convert_airline_calculate,
    "cancel_reservation": convert_airline_cancel_reservation,
    "update_reservation_baggages": convert_airline_update_reservation_baggages,
    "update_reservation_flights": convert_airline_update_reservation_flights,
    "update_reservation_passengers": convert_airline_update_reservation_passengers,
    "send_certificate": convert_airline_send_certificate,
    "book_reservation": convert_airline_book_reservation,
}

SKIP_TOOLS = {"think", "transfer_to_human_agents"}


def detect_domain_from_tools(tool_calls: Sequence[ToolCall]) -> str:
    return "airline"


def parse_dialog(dialog: Dict[str, Any]) -> Dict[str, Any]:
    conversations = dialog.get("conversations") or []
    first_user: Optional[str] = None
    human_messages: List[str] = []
    tool_calls: List[ToolCall] = []
    last_observed_user_id: Optional[str] = None  # Track user_id for heuristic injection

    for idx, message in enumerate(conversations):
        sender = message.get("from")
        if sender == "human":
            text = message.get("value", "")
            human_messages.append(text)
            if first_user is None:
                first_user = text
        if sender == "function_call":
            payload = safe_json_loads(message.get("value", "")) or {}
            name = payload.get("name")
            arguments = payload.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = safe_json_loads(arguments) or {}
            
            obs_obj: Any = None
            if idx + 1 < len(conversations) and conversations[idx + 1].get("from") == "observation":
                obs_obj = safe_json_loads(conversations[idx + 1].get("value", ""))

            # Heuristic: Inject user_id if missing and available from context (fix for model laziness)
            if name in ["cancel_reservation", "update_reservation_baggages", "update_reservation_flights", "update_reservation_passengers"]:
                if isinstance(arguments, dict) and "user_id" not in arguments and last_observed_user_id:
                    arguments["user_id"] = last_observed_user_id

            # Update context user_id from arguments or observation
            if isinstance(arguments, dict) and "user_id" in arguments:
                last_observed_user_id = str(arguments["user_id"])
            if isinstance(obs_obj, dict) and "user_id" in obs_obj:
                last_observed_user_id = str(obs_obj["user_id"])

            tool_calls.append(
                ToolCall(
                    tool_name=name,
                    arguments=arguments,
                    observation=obs_obj,
                    human_messages=tuple(human_messages),
                )
            )
    return {
        "first_user": first_user or DEFAULT_FIRST_USER,
        "tool_calls": tool_calls,
        "all_human_messages": human_messages,
    }


def convert_tool_calls(
    tool_calls: Sequence[ToolCall],
    conversation_index: int,
    min_actions: int,
    domain_config: DomainConfig,
) -> Tuple[List[SqlAction], List[SqlAction]]:
    sql_actions: List[SqlAction] = []
    for action_index, call in enumerate(tool_calls):
        if not call.tool_name or call.tool_name in SKIP_TOOLS:
            continue
        handler = domain_config.converters.get(call.tool_name)
        action: Optional[SqlAction]
        if handler is None:
            action = domain_config.fallback_converter(call, {"conversation_index": conversation_index, "action_index": action_index})
        else:
            meta = {"conversation_index": conversation_index, "action_index": action_index}
            action = handler(call, meta)
        if action:
            action.tool_call_index = action_index
            action.source_action_index = len(sql_actions)
            sql_actions.append(action)
    if len(sql_actions) < min_actions:
        return [], []
    # Keep each tool call's SQL bundle intact (no statement splitting).
    return list(sql_actions), list(sql_actions)


def build_instruction(first_user: str, actions: Sequence[SqlAction]) -> str:
    lines = [first_user.strip() or DEFAULT_FIRST_USER]
    if actions:
        lines.append("")
        lines.append("You should perform and let the assistant complete these database operations in order:")
        for idx, action in enumerate(actions, start=1):
            lines.append(f"{idx}. {action.description}")
    return "\n".join(lines).strip()


def build_ground_truth(
    actions: Sequence[SqlAction],
    db_path: Optional[str],
    expsem_for_readonly: Optional[str] = None,
) -> Dict[str, Any]:
    gt_actions = [{"name": "sqlexe", "arguments": {"sql": action.sql}} for action in actions]
    outputs: List[Any] = []
    result_hashes: List[Dict[str, Any]] = []

    if not db_path:
        raise ValueError("ground truth generation requires a database path")

    tmp_path: Optional[str] = None
    conn = None
    tmp_path, conn, err = _open_temp_conn(db_path)
    if err or not conn or not tmp_path:
        raise RuntimeError(f"Failed to open temp database for ground-truth execution: {err}")

    try:
        for action in actions:
            payload: Optional[Dict[str, Any]] = None
            sql_text = action.sql or ""
            try:
                result = _exec_and_hash_on_conn(
                    conn,
                    sql_text,
                    max_rows=MAX_ROWS,
                    verify_writes=True,
                )
            except Exception as exc:  # pragma: no cover - conversion diagnostic
                logging.getLogger(__name__).warning(
                    "Failed to execute SQL action for ground truth: %s", exc
                )
                result = {"payload": None, "error": str(exc)}

            candidate = result.get("payload")
            include_payload = False
            if isinstance(candidate, dict):
                payload_type = candidate.get("type")
                if payload_type == "error":
                    include_payload = True
                elif payload_type == "rows" and is_read_query(sql_text):
                    include_payload = True
            if include_payload:
                payload = json.loads(json.dumps(candidate, ensure_ascii=False))
            outputs.append(payload)

            db_hash = _file_sha256(tmp_path)
            result_hashes.append(
                {
                    "sql_key": _sql_key(sql_text) if sql_text else None,
                    "hash": db_hash,
                }
            )
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    ground_truth = {
        "actions": gt_actions,
        "outputs": outputs,
        "style": "rule",
        "expsem_for_readonly": expsem_for_readonly,
    }
    digest = hashlib.sha256(json.dumps(ground_truth, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    ground_truth["gt_data_hash"] = digest
    ground_truth["result_hashes"] = result_hashes
    return {"ground_truth": ground_truth}


def build_user_prompt(first_user: str, instruction: str) -> List[Dict[str, str]]:
    intro = _first_turn_utterance(first_user)
    system_prompt = USER_SIM_PROMPT_TEMPLATE.format(instruction=instruction)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Hi! How can I help you today?",
        },
        # Purposefully omit the simulated user's reply so the model needs to
        # produce it; otherwise it continues the conversation as the agent.
    ]


def build_prompt(first_user: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SQL_AGENT_POLICY},
        {"role": "user", "content": _first_turn_utterance(first_user)},
    ]


def build_extra_info(
    idx: int,
    dialog: Dict[str, Any],
    actions: Sequence[SqlAction],
    tool_calls: Sequence[ToolCall],
    db_name: str,
    db_path: Optional[str],
    schema_path: Optional[str],
) -> Dict[str, Any]:
    tools_kwargs = None
    if db_path or schema_path:
        sqlexe_kwargs: Dict[str, Any] = {"db_name": db_name}
        get_schema_kwargs: Dict[str, Any] = {"db_name": db_name}
        if db_path:
            sqlexe_kwargs["db_path"] = db_path
            get_schema_kwargs["db_path"] = db_path
        if schema_path:
            get_schema_kwargs["schema_path"] = schema_path
        tools_kwargs = {
            "sqlexe": {"create_kwargs": sqlexe_kwargs},
            "getdbschema": {"create_kwargs": get_schema_kwargs},
        }
    action_index_map: Dict[int, List[int]] = {}
    for action_index, action in enumerate(actions):
        if action.tool_call_index is None:
            continue
        action_index_map.setdefault(action.tool_call_index, []).append(action_index)
    extra = {
        "conversation_index": idx,
        "need_tools_kwargs": tools_kwargs is not None,
        "tools_kwargs": tools_kwargs,
        "raw_tool_calls": [
            {
                "tool_call_index": call_index,
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "observation": call.observation,
                "action_indices": action_index_map.get(call_index, []),
                "action_count": len(action_index_map.get(call_index, [])),
            }
            for call_index, call in enumerate(tool_calls)
        ],
        "source_dialog": dialog.get("conversations"),
    }
    if schema_path:
        extra["schema_path"] = schema_path
    if db_path:
        extra["db_path"] = db_path
    return extra


def build_record(
    idx: int,
    dialog: Dict[str, Any],
    ability: str,
    data_source: str,
    db_name: str,
    db_path: Optional[str],
    schema_path: Optional[str],
    instruction: str,
    first_user: str,
    actions: Sequence[SqlAction],
    tool_calls: Sequence[ToolCall],
) -> Dict[str, Any]:
    return {
        "ability": ability,
        "data_source": data_source,
        "db_name": db_name,
        "db_path": db_path,
        "schema_path": schema_path,
        "extra_info": build_extra_info(idx, dialog, actions, tool_calls, db_name, db_path, schema_path),
        "prompt": build_prompt(first_user),
        "user_prompt": build_user_prompt(first_user, instruction),
        "question": instruction,
        "reward_model": build_ground_truth(actions, db_path, expsem_for_readonly=None),
    }


def convert_dataset(
    input_path: Path,
    domain_config: DomainConfig,
    min_actions: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as infile:
        dialogs = json.load(infile)
    if limit is not None:
        dialogs = dialogs[:limit]
    records: List[Dict[str, Any]] = []
    skipped = 0
    for idx, dialog in enumerate(dialogs):
        parsed = parse_dialog(dialog)
        # Domain fixed to airline.
        actions, original_actions_list = convert_tool_calls(parsed["tool_calls"], idx, min_actions, domain_config)
        if not actions:
            skipped += 1
            continue
        
        rewritten_instruction = dialog.get("instruction")
        original_actions_str = build_instruction(parsed["first_user"], original_actions_list)

        if rewritten_instruction:
            parts = original_actions_str.split("\n", 1)
            original_steps = parts[1] if len(parts) > 1 else ""
            instruction = f"{rewritten_instruction}\n\n{original_steps}".strip()
        else:
            instruction = original_actions_str

        record = build_record(
            idx=idx,
            dialog=dialog,
            ability=domain_config.ability,
            data_source=domain_config.data_source,
            db_name=domain_config.db_name,
            db_path=domain_config.db_path,
            schema_path=domain_config.schema_path,
            instruction=instruction,
            first_user=parsed["first_user"],
            actions=actions,
            tool_calls=parsed["tool_calls"],
        )
        records.append(record)
    logging.info("Converted %d dialogs into %d records (skipped %d)", len(dialogs), len(records), skipped)
    return records


def write_output(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if not records:
        raise RuntimeError("No records were generated; nothing to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json as _json

    normalised: List[Dict[str, Any]] = []
    for r in records:
        nr = r.copy()
        # Normalise extra_info.raw_tool_calls -> list<struct<tool_name,arguments,observation>> with JSON strings
        ei = nr.get("extra_info")
        if isinstance(ei, dict):
            rtc = ei.get("raw_tool_calls")
            if isinstance(rtc, list):
                new_calls: List[Dict[str, Any]] = []
                for c in rtc:
                    if not isinstance(c, dict):
                        continue
                    new_calls.append(
                        {
                            "tool_call_index": c.get("tool_call_index"),
                            "action_indices": c.get("action_indices"),
                            "action_count": c.get("action_count"),
                            "tool_name": c.get("tool_name"),
                            "arguments": _json.dumps(c.get("arguments"), ensure_ascii=False, sort_keys=True) if c.get("arguments") is not None else None,
                            "observation": _json.dumps(c.get("observation"), ensure_ascii=False) if c.get("observation") is not None else None,
                        }
                    )
                ei["raw_tool_calls"] = new_calls
            # Normalise source_dialog list to struct list with only required keys
            sd = ei.get("source_dialog")
            if isinstance(sd, list):
                ei["source_dialog"] = [
                    {"from": m.get("from"), "value": m.get("value")}
                    for m in sd if isinstance(m, dict)
                ]
        # Ensure prompt & user_prompt entries each dict has content & role keys (already built that way)
        for key in ("prompt", "user_prompt"):
            seq = nr.get(key)
            if isinstance(seq, list):
                cleaned = []
                for msg in seq:
                    if not isinstance(msg, dict):
                        continue
                    cleaned.append({"content": msg.get("content"), "role": msg.get("role")})
                nr[key] = cleaned
        # Reward model: normalise actions arguments to have only sql key (already) but keep dict nesting
        rm = nr.get("reward_model")
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")
            if isinstance(gt, dict):
                actions = gt.get("actions")
                if isinstance(actions, list):
                    new_actions = []
                    for a in actions:
                        if not isinstance(a, dict):
                            continue
                        args = a.get("arguments")
                        if isinstance(args, dict):
                            args = {"sql": args.get("sql")}
                        new_actions.append({"name": a.get("name"), "arguments": args})
                    gt["actions"] = new_actions
                outputs = gt.get("outputs")
                if isinstance(outputs, list):
                    serialised_outputs = []
                    for item in outputs:
                        if item is None:
                            serialised_outputs.append(None)
                        else:
                            serialised_outputs.append(_json.dumps(item, ensure_ascii=False, sort_keys=True))
                    gt["outputs"] = serialised_outputs
        normalised.append(nr)

    table = pa.Table.from_pylist(normalised)
    pq.write_table(table, output_path)
    logging.info("Wrote %d rows (nested) to %s", len(normalised), output_path)


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)
    input_path = Path(args.input)
    output_path = Path(args.output)

    def _resolve_absolute(path_str: Optional[str]) -> Optional[Path]:
        if not path_str:
            return None
        candidate = Path(path_str).expanduser()
        if not candidate.is_absolute():
            candidate = (WORKSPACE_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    def _resolve_with_default(value: Optional[str], default_path: Path) -> Optional[Path]:
        path_candidate = value if value is not None else str(default_path)
        if not path_candidate:
            return None
        return _resolve_absolute(path_candidate)

    # Airline domain setup
    airline_ability = args.ability or "sqlbench_apigen_airline"
    airline_data_source = args.data_source or "sqlbench_apigen_airline"
    airline_db_name = args.db_name or airline_data_source
    airline_db_abs = _resolve_with_default(args.db_path, DEFAULT_AIRLINE_DB_PATH)
    airline_schema_abs = _resolve_with_default(args.schema_path, DEFAULT_AIRLINE_SCHEMA_PATH)
    if airline_schema_abs is None and airline_db_abs is not None:
        fallback_schema = airline_db_abs.with_name("schema.sql")
        if fallback_schema.exists():
            airline_schema_abs = fallback_schema
    if airline_db_abs is not None and not airline_db_abs.exists():
        logging.warning("Airline database path %s does not exist; embedding reference anyway.", airline_db_abs)
    if airline_schema_abs is None:
        logging.info("Airline schema path missing; getdbschema will rely on introspection.")
    elif not airline_schema_abs.exists():
        logging.warning("Airline schema path %s does not exist; tools may need introspection.", airline_schema_abs)

    airline_config = DomainConfig(
        ability=airline_ability,
        data_source=airline_data_source,
        db_name=airline_db_name,
        db_path=str(airline_db_abs.resolve()) if airline_db_abs else None,
        schema_path=str(airline_schema_abs.resolve()) if airline_schema_abs else None,
        converters=AIRLINE_TOOL_CONVERTERS,
        fallback_converter=convert_airline_fallback,
    )

    # Airline only: no retail configuration.

    records = convert_dataset(
        input_path=input_path,
        domain_config=airline_config,
        min_actions=args.min_actions,
        limit=args.limit,
    )
    write_output(records, output_path)


if __name__ == "__main__":
    main()
