# utils/helper.py
import os
import json
import re
from typing import Any, Dict


def formatContext(retrieved_elements):
    return "\n".join(doc.metadata["text"] for doc in retrieved_elements['matches'])

# Ensure text is in UTF-8 bytes
def to_utf8_bytes(text):
    if isinstance(text, bytes):
        return text
    return text.encode("utf-8", errors="replace")

def load_codebase_metadata():
    global dir_structure, call_graph

    if not os.path.exists("database/dir_structure.json") or not os.path.exists("database/call_graph.json"):
        raise RuntimeError("Codebase not indexed yet. Please parse repository first.")

    with open("database/dir_structure.json", "r") as f:
        dir_structure = f.read()

    with open("database/call_graph.json", "r") as f:
        call_graph = f.read()

def to_json_safe(text: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if default is None:
        default = {}

    if not text:
        return default

    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return default
