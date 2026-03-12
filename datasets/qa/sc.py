import json
from pathlib import Path

# -------------------
# CONFIG
# -------------------
INPUT_PATH = "medarabiq.json"      # or .jsonl
OUTPUT_PATH = "medarabiq-text.json"  # or .jsonl
NEW_FIELD = "answer_text"         # name of the new field to add

# If your dataset uses Arabic letters sometimes, map them too.
LETTER_MAP = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F",
    "أ": "A", "ا": "A",
    "ب": "B",
    "ج": "C",
    "د": "D",
    "هـ": "E", "ه": "E", "ة": "E",
    "و": "F",  # optional, only if you ever use it for option F
}

OPTION_KEY_BY_LETTER = {
    "A": "opa",
    "B": "opb",
    "C": "opc",
    "D": "opd",
    "E": "ope",
    "F": "opf",
}

def normalize_answer_letter(x) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    return LETTER_MAP.get(x, x)

def option_text_for(item: dict, letter: str) -> str:
    key = OPTION_KEY_BY_LETTER.get(letter, "")
    if not key:
        return ""
    val = item.get(key, "")
    return "" if val is None else str(val).strip()

def process_items(items):
    missing = 0
    for it in items:
        letter = normalize_answer_letter(it.get("answer", ""))
        txt = option_text_for(it, letter)
        it[NEW_FIELD] = txt
        if not txt:
            missing += 1
    return missing

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def save_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    in_path = Path(INPUT_PATH)
    out_path = Path(OUTPUT_PATH)

    if in_path.suffix.lower() == ".jsonl":
        items = load_jsonl(in_path)
        missing = process_items(items)
        save_jsonl(out_path, items)
    else:
        data = load_json(in_path)
        if not isinstance(data, list):
            raise ValueError("Expected INPUT .json to be a list of items at the top level.")
        missing = process_items(data)
        save_json(out_path, data)

    print(f"Done. Wrote: {out_path}")
    print(f"Items processed: {len(items) if in_path.suffix.lower()=='.jsonl' else len(data)}")
    print(f"Missing/empty {NEW_FIELD}: {missing}")

if __name__ == "__main__":
    main()