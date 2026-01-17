# utils/helper.py

# Formatting the retrieved elements into a string context
import os


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
