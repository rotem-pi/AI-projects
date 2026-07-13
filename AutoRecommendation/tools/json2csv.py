"""Convert JSON on stdin to CSV at argv[1].

- list of objects  -> one row per object, columns = union of keys
- single object    -> two columns: key,value (nested values JSON-encoded)
- other JSON       -> single "value" column
Nested/complex cell values are serialized back to compact JSON.
"""

import csv
import json
import sys


def cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def main() -> int:
    out_path = sys.argv[1]
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 1

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
