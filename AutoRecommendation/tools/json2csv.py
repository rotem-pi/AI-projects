"""Convert JSON to CSV.

    python tools/json2csv.py out.csv < response.json

- list of objects  -> one row per object, columns = union of keys
- single object    -> two columns: key,value (nested values JSON-encoded)
- other JSON       -> single "value" column
Nested/complex cell values are serialized back to compact JSON.

write_csv() is the library form used by tools/dump_rest_api.py.
"""

import csv
import json
import sys
from pathlib import Path


def cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_csv(data: object, out_path: Path | str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            columns: list[str] = []
            seen: set[str] = set()
            for item in data:
                for key in item:
                    if key not in seen:
                        seen.add(key)
                        columns.append(key)
            writer.writerow(columns)
            for item in data:
                writer.writerow([cell(item.get(col)) for col in columns])
        elif isinstance(data, dict):
            writer.writerow(["key", "value"])
            for key, value in data.items():
                writer.writerow([key, cell(value)])
        else:
            writer.writerow(["value"])
            if isinstance(data, list):
                for item in data:
                    writer.writerow([cell(item)])
            else:
                writer.writerow([cell(data)])


def main() -> int:
    out_path = sys.argv[1]
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 1
    write_csv(data, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
