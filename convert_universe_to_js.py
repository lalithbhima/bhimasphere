import json
from pathlib import Path

# Input and output paths
json_path = Path("reports/step5_universe.json")
js_path = Path("nasa-exo-ui/js/data/exoplanets.js")

# Load JSON
data = json.loads(json_path.read_text())

# Write as JS export
with open(js_path, "w") as f:
    f.write("export const exoplanets = ")
    json.dump(data, f, indent=2)
    f.write(";\n")

print(f"✅ Converted {json_path} → {js_path}")
