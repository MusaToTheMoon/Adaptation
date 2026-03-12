import pandas as pd
import json

CSV_PATH = "train.csv"
OUTPUT_JSON = "train.json"

df = pd.read_csv(CSV_PATH).fillna("")

df = df.rename(columns={
    "Question Number": "id",
    "Question": "question",
    "Option A": "opa",
    "Option B": "opb",
    "Option C": "opc",
    "Option D": "opd",
    "Option E": "ope",
    "Option F": "opf",
    "Correct Answer": "answer",
})

keep_cols = [
    "id",
    "question",
    "opa", "opb", "opc", "opd", "ope", "opf",
    "answer"
]

df = df[keep_cols]

# Ensure everything is string
for col in keep_cols:
    df[col] = df[col].astype(str).str.strip()

# Normalize answer to A-F
df["answer"] = df["answer"].str.upper().str.extract(r"([A-F])", expand=False)
df = df[df["answer"].isin(list("ABCDEF"))]

records = df.to_dict(orient="records")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(records)} records to {OUTPUT_JSON}")