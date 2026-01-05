import os
import shutil
import subprocess
import sys
from git import Repo
import mimetypes
from tree_sitter_language_pack import get_parser
import json

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
    
def extract_called_function(call_node):
    fn = call_node.child_by_field_name("function")
    if not fn:
        return None

    if fn.type == "identifier":
        return fn.text.decode("utf-8")

    if fn.type == "attribute":
        obj = fn.child(0)
        attr = fn.child(2)
        if obj and attr:
            return f"{obj.text.decode('utf-8')}.{attr.text.decode('utf-8')}"

    return None

# Recursive AST Walk to extract functions and classes
def walk(node, file, code, function_index, call_graph, current_function=None):
    chunk = []

    # Function Definition
    if node.type == "function_definition":
        name = node.child_by_field_name("name").text.decode("utf-8")
        fq_name = f"{file['path']}::{name}"

        function_index[name] = fq_name
        call_graph.setdefault(fq_name, set())

        current_function = fq_name

        chunk.append({
            "id": fq_name,
            "code": code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
            "metadata": {
                "type": "function",
                "language": file["file_type"],
                "file": file["path"],
                "start_line" : node.start_point[0]+1,
                "end_line" : node.end_point[0]+1,
            }
        })

    # Class Definition
    elif node.type == "class_definition":
        name = node.child_by_field_name("name").text.decode("utf-8")
        fq_name = f"{file['path']}::{name}"

        chunk.append({
            "id": fq_name,
            "code": code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
            "metadata": {
                "type": "class",
                "language": file["file_type"],
                "file": file["path"],
                "start_line" : node.start_point[0]+1,
                "end_line" : node.end_point[0]+1,
            }
        })

    # Function Call
    elif node.type == "call" and current_function:
        called = extract_called_function(node)
        if called:
            call_graph[current_function].add(called)

    # Recurse
    for child in node.children:
        chunk.extend(
            walk(
                child,
                file,
                code,
                function_index,
                call_graph,
                current_function
            )
        )

    return chunk


# Extract code chunks from repository habe to add an additional argumet for the repo link, to clone it locally and delete after use
def extract_chunks(language: str = "python"):
    local_path = "./public"
    repo_url = "https://github.com/Inhuman-Jester/RAG-Project.git"
    Repo.clone_from(repo_url, local_path)
    repo = Repo(local_path)

    files = [
        item.path
        for item in repo.tree().traverse()
        if item.type == "blob"
    ]
    print(f"Found {len(files)} files in the repository.")

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

        print(f"Read file: {f} (type: {file_type})")

    parser = get_parser(language)

    chunks = []
    function_index = {}
    raw_call_graph = {}

    # Parse each file
    for f in code_files:
        code_bytes = to_utf8_bytes(f["content"])
        tree = parser.parse(code_bytes)
        root = tree.root_node

        chunks.extend(
            walk(
                root,
                f,
                code_bytes,
                function_index,
                raw_call_graph
            )
        )
        print(f"Parsed file: {f['path']}")
        
    # Resolve Call Graph
    call_graph = {}

    for caller, callees in raw_call_graph.items():
        call_graph[caller] = list()

        for callee in callees:
            if callee in function_index:
                call_graph[caller].append(function_index[callee])
            else:
                call_graph[caller].append(callee)  # unresolved / external
    
    json.dump(call_graph, open('database/call_graph.json', 'w'), indent=2)
    
    def delete_after_exit(path):
        subprocess.Popen(
            [
                "cmd", "/c",
                f"timeout /t 1 >nul && rmdir /s /q {path} && mkdir {path}"
            ],
            creationflags=subprocess.CREATE_NO_WINDOW
        )

    delete_after_exit("public")

    return chunks

if __name__ == "__main__":
    chunks = extract_chunks(language="python")
    print(f"Extracted {len(chunks)} code chunks from the repository.")