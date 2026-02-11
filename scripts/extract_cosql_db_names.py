#!/usr/bin/env python3
"""Extract unique db_name (db_id) values from COSQL dialogs JSON.

Usage:
  python extract_cosql_db_names.py \
    COSQL/cosql_dataset/cosql_all_info_dialogs.json \
      --out db_names.txt

The input file format (cosql_all_info_dialogs.json) is a large JSON object where
keys are dialog ids and values are objects containing at least a "db_id" field.
Some downstream parquet rows may use the key name "db_name"; we normalize both.

Output:
  Writes a newline-delimited sorted list of unique database names to the output
  path (default: db_names.txt) and also prints them to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Set


def load_json(path: Path):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[ERROR] JSON decode failed: {e}\n")
        raise
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[ERROR] Unable to read file {path}: {e}\n")
        raise


def extract_db_names(obj) -> Set[str]:
    names: Set[str] = set()
    if isinstance(obj, dict):
        # COSQL root: { dialog_id: dialog_obj, ... }
        for _dialog_id, dialog in obj.items():
            if not isinstance(dialog, dict):
                continue
            # Primary key in original dataset appears as 'db_id'
            v = dialog.get('db_id') or dialog.get('db_name')
            if isinstance(v, str) and v.strip():
                names.add(v.strip())
            # Some nested structures might store different naming; scan shallowly
            for k, vv in dialog.items():
                if k in ('db_id', 'db_name') and isinstance(vv, str) and vv.strip():
                    names.add(vv.strip())
    elif isinstance(obj, list):
        # If format changes to list of dialog objects
        for dialog in obj:
            if not isinstance(dialog, dict):
                continue
            v = dialog.get('db_id') or dialog.get('db_name')
            if isinstance(v, str) and v.strip():
                names.add(v.strip())
    return names


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract unique COSQL db names")
    parser.add_argument('input', type=Path, help='Path to cosql_all_info_dialogs.json')
    parser.add_argument('--out', type=Path, default=Path('db_names.txt'), help='Output file path')
    parser.add_argument('--show-count', action='store_true', help='Print count summary to stderr')
    args = parser.parse_args(argv)

    data = load_json(args.input)
    db_names = extract_db_names(data)
    sorted_names = sorted(db_names)

    # Write output
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('w', encoding='utf-8') as f:
            for name in sorted_names:
                f.write(name + '\n')
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[ERROR] Failed writing output file {args.out}: {e}\n")
        return 2

    # Print to stdout for immediate visibility
    for name in sorted_names:
        print(name)

    if args.show_count:
        sys.stderr.write(f"[INFO] Extracted {len(sorted_names)} unique db names.\n")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
