# utils.py
from typing import Dict
import json

def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)