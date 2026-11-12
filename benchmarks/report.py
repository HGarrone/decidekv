"""
Tables of the README and of the article, generated from the result files.

`benchmarks/tables.json` describes every table: literal cells, and cells that point to a number in `results/`.
Each document holds markers `<!-- table:NAME -->` ... `<!-- /table:NAME -->`; the text between them is replaced.

    python benchmarks/report.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MISSING = "–"


def number(value, kind):
    if value is None:
        return MISSING
    if kind in ("acc", "pct"):
        return f"{100 * value:.1f}%"
    elif kind == "ms":
        return f"{value:,.0f} ms"
    elif kind == "s":
        return f"{value:.0f} s"
    elif kind == "int":
        return f"{int(value):,}"
    elif kind == "text":
        return str(value)
    else:
        raise ValueError(f"Unknown number kind: {kind}")


def render(cell, results, suffix):
    if isinstance(cell, list):
        return "".join(render(part, results, suffix) for part in cell)
    if not isinstance(cell, str) or not cell.startswith("@"):
        return cell
    name, rest = cell[1:].split("#", 1)
    key, show = rest.rsplit("|", 1)
    file_path = os.path.join(ROOT, "results", f"{name}_{suffix}.json")
    if file_path not in results:
        try:
            with open(file_path, encoding="utf8") as f:
                results[file_path] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results[file_path] = {}
    row = results[file_path].get(key)
    if row is None or not isinstance(row, dict):
        return MISSING
    if show == "acc_ms":
        acc = row.get("acc")
        ms = row.get("ms")
        if acc is None:
            return MISSING
        elif ms is None:
            return number(acc, "acc")
        else:
            return number(acc, "acc") + " · " + number(ms, "ms")
    else:
        colon_pos = show.find(":")
        if colon_pos != -1:
            kind, field = show.split(":", 1)
            return number(row.get(field), kind)
        elif show == "acc":
            return number(row.get("acc"), "acc")
        elif show == "ms":
            return number(row.get("ms"), "ms")
        else:
            raise ValueError(f"Unknown show type: {show}")


def table(spec, results, suffix):
    header = spec["header"]
    rows = spec["rows"]
    rendered_header = " | ".join(render(cell, results, suffix) for cell in header)
    result_lines = [f"| {rendered_header} |"]
    result_lines.append("|" + "---|" * len(header))
    for row in rows:
        rendered_row = " | ".join(render(cell, results, suffix) for cell in row)
        result_lines.append(f"| {rendered_row} |")
    return "\n".join(result_lines)


def main():
    tables_path = os.path.join(ROOT, "benchmarks", "tables.json")
    with open(tables_path, encoding="utf8") as f:
        tables_data = json.load(f)
    suffix = tables_data["suffix"]
    documents = tables_data["documents"]
    results = {}
    table_dict = {}
    for name, spec in tables_data["tables"].items():
        table_dict[name] = table(spec, results, suffix)
    for path in documents:
        full_path = os.path.join(ROOT, path)
        if not os.path.exists(full_path):
            print(f"{path}: not found")
            continue
        with open(full_path, encoding="utf8") as f:
            text = f.read()
        count = 0
        for name, markdown in table_dict.items():
            pattern = re.compile(
                r"(<!-- table:" + re.escape(name) + r" -->\n).*?(<!-- /table:" + re.escape(name) + r" -->)",
                re.S,
            )
            text, n = pattern.subn(
                lambda m: m.group(1) + markdown + "\n" + m.group(2), text
            )
            count += n
        if count > 0:
            with open(full_path, "w", encoding="utf8", newline="\n") as f:
                f.write(text)
        print(f"{path}: {count} tables updated")


if __name__ == "__main__":
    main()
