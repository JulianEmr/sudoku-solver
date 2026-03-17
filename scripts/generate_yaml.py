import json
import yaml
import os

NOTES_PATH = "data/label-studio/notes.json"
OUTPUT_DIR = "data"

def get_classes_from_notes(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    categories = data.get('categories', [])   
    class_names = [cat['name'] for cat in categories]
    return class_names

classes = get_classes_from_notes(NOTES_PATH)
print(f"Extracted classes: {classes}")

data_config = {
    'path': os.path.relpath(OUTPUT_DIR), 
    'train': 'train/images',
    'val': 'val/images',
    'nc': len(classes),
    'names': {i: name for i, name in enumerate(classes)}
}

yaml_file = os.path.join(OUTPUT_DIR, "dataset.yaml")
with open(yaml_file, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"Success! YOLO config saved to: {yaml_file}")