import json
with open("split_data/val/annotations/instances_val.json") as f:
    data = json.load(f)

assert "info" in data, "Missing 'info' block!"
assert "images" in data, "Missing 'images' list!"
assert "annotations" in data, "Missing 'annotations' list!"
assert "categories" in data, "Missing 'categories' list!"
print("✅ Your JSON is good!")
