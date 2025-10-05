import pandas as pd
import os
import io

# === List of NASA exoplanet CSV datasets ===
datasets = [
    "kepler_objects_of_interest",
    "k2_planets_and_candidates",
    "TESS_objects_of_interest",
]

def smart_read_csv(path: str) -> pd.DataFrame:
    """Read CSV safely, skipping '#' headers and guessing delimiter."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 1️⃣ Skip leading comment lines starting with '#'
    clean_lines = [line for line in lines if not line.startswith("#")]

    # 2️⃣ Join them into an in-memory buffer for pandas
    buffer = io.StringIO("".join(clean_lines))

    # 3️⃣ Try multiple delimiters (comma, tab, semicolon)
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(buffer, sep=sep)
            if len(df.columns) > 1:
                print(f"✅ Detected delimiter '{sep}' with {len(df.columns)} columns")
                return df
        except Exception:
            buffer.seek(0)
            continue

    raise ValueError(f"❌ Could not parse {path} with common delimiters")

# === Main loop ===
for name in datasets:
    csv_path = f"{name}.csv"
    json_path = f"{name}.json"

    if not os.path.exists(csv_path):
        print(f"⚠️  Skipping missing file: {csv_path}")
        continue

    print(f"📂 Reading {csv_path} ...")
    df = smart_read_csv(csv_path)
    print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # 4️⃣ Optional: drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # 5️⃣ Export to JSON list-of-dicts format
    df.to_json(json_path, orient="records", indent=2)
    print(f"💾 Saved → {json_path}\n")

print("🎉 All datasets converted successfully!")
