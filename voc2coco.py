import os
import json
import xml.etree.ElementTree as ET

from tqdm import tqdm

def convert(xml_dir, output_json_path, image_dir, label_map):
    output = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {},  # <- this avoids the KeyError: 'info'
        "licenses": []
    }

    annotation_id = 1
    image_id = 1

    for i, class_name in enumerate(label_map.values()):
        output["categories"].append({
            "id": i + 1,
            "name": class_name,
            "supercategory": "none"
        })

    for xml_file in tqdm(os.listdir(xml_dir)):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()

        filename = root.find("filename").text
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        output["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall("object"):
            label = obj.find("name").text
            category_id = list(label_map.values()).index(label) + 1

            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
            w = xmax - xmin
            h = ymax - ymin

            output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=4)

# Example usage:
xml_dir = "split_data/val/annotations"
image_dir = "split_data/val/images"
output_json_path = "split_data/val/val_coco.json"
label_map = {1: "cone"}  # match your label map

convert(xml_dir, output_json_path, image_dir, label_map)
