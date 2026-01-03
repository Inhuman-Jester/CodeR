import os
from git import Repo
import mimetypes
from tree_sitter_language_pack import get_parser

# Ensure text is in UTF-8 bytes
def to_utf8_bytes(text):
    if isinstance(text, bytes):
        return text
    return text.encode("utf-8", errors="replace")

# Read file content from git repository
def read_file(repo, path):
    try:
        return repo.git.show(f'HEAD:{path}')
    except Exception:
        return None

# Recursive AST Walk to extract functions and classes
def walk(node, file, code):
    chunk = []

    if node.type == "function_definition":
        name = node.child_by_field_name("name").text.decode("utf-8")

        chunk.append({
            "id": name,
            "code": code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
            "metadata": {
                "type": "function",
                "language": file["file_type"],
                "file": file["path"],
                "dependencies": [],  # fill later via static analysis if needed
                "line_of_code": [
                    node.start_point[0] + 1,
                    node.end_point[0] + 1
                ]
            }
        })

    elif node.type == "class_definition":
        name = node.child_by_field_name("name").text.decode("utf-8")

        chunk.append({
            "id": name,
            "code": code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
            "metadata": {
                "type": "class",
                "language": file["file_type"],
                "file": file["path"],
                "dependencies": [],
                "line_of_code": [
                    node.start_point[0] + 1,
                    node.end_point[0] + 1
                ]
            }
        })

    for child in node.children:
        chunk.extend(walk(child, file, code))

    return chunk

# Extract code chunks from repository habe to add an additional argumet for the repo link, to clone it locally and delete after use
def extract_chunks(
    language: str = "python"
):
    local_path="./public"
    repo = Repo(local_path)

    files = [
        item.path
        for item in repo.tree().traverse()
        if item.type == "blob"
    ]

    code_files = []

    for f in files:
        content = read_file(repo, f)
        _, ext = os.path.splitext(f)

        mime = mimetypes.guess_type(f)[0]
        file_type = mime.split("/")[-1] if mime else ext[1:]

        if content:
            code_files.append({
                "path": f,
                "content": content,
                "file_type": file_type
            })

    parser = get_parser(language)
    chunks = []

    for f in code_files:
        code_bytes = to_utf8_bytes(f["content"])
        tree = parser.parse(code_bytes)
        root = tree.root_node

        chunks.extend(walk(root, f, code_bytes))

    return chunks
