import json

json_path = "split_data/val/annotations/instances_val.json"

with open(json_path, "r") as f:
    data = json.load(f)

# Add "info" if it's missing
if "info" not in data:
    data["info"] = {
        "description": "Cone Dataset",
        "version": "1.0",
        "contributor": "jtristan",
        "date_created": "2025-07-28"
    }

# Save it back
with open(json_path, "w") as f:
    json.dump(data, f)

print("✅ 'info' block injected successfully.")
