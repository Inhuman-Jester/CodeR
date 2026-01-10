import os
import subprocess
from git import Repo
import mimetypes
from tree_sitter_language_pack import get_parser
import json
import logging
from utils.helper import to_utf8_bytes

# Read file content from git repository
def read_file(repo, path):
    try:
        return repo.git.show(f'HEAD:{path}')
    except Exception as e:
        logging.warning(f"Failed to read file {path}: {e}")
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
def walk(node, file, code, function_index, call_graph, file_struct, current_function=None):
    space = []

    # Function Definition
    if node.type == "function_definition":
        name = node.child_by_field_name("name").text.decode("utf-8")
        fq_name = f"{file['path']}::{name}"

        logging.debug(f"Discovered function: {fq_name}")

        function_index[name] = fq_name
        call_graph.setdefault(fq_name, set())

        current_function = fq_name

        space.append({
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
        file_struct["functions"].append(fq_name)

    # Class Definition
    elif node.type == "class_definition":
        name = node.child_by_field_name("name").text.decode("utf-8")
        fq_name = f"{file['path']}::{name}"

        logging.debug(f"Discovered class: {fq_name}")

        space.append({
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
        file_struct["classes"].append(fq_name)

    # Function Call
    elif node.type == "call" and current_function:
        called = extract_called_function(node)
        if called:
            logging.debug(f"Function call detected: {current_function} -> {called}")
            call_graph[current_function].add(called)

    # Recurse
    for child in node.children:
        space.extend(
            walk(
                child,
                file,
                code,
                function_index,
                call_graph,
                file_struct,
                current_function,
            )
        )

    return space


# Extract code chunks from repository habe to add an additional argumet for the repo link, to clone it locally and delete after use
def extract_spaces(repo_url, language: str = "python"):
    local_path = "./public"

    logging.info(f"Cloning repository from {repo_url} into {local_path}")
    Repo.clone_from(repo_url, local_path)

    repo = Repo(local_path)

    files = [
        item.path
        for item in repo.tree().traverse()
        if item.type == "blob"
    ]

    logging.info(f"Discovered {len(files)} files in repository")

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

    logging.info(f"Filtered {len(code_files)} readable code files")

    parser = get_parser(language)

    spaces = {}
    function_index = {}
    raw_call_graph = {}
    dir_structure = {}

    # Parse each file
    for f in code_files:
        logging.debug(f"Parsing file: {f['path']}")

        code_bytes = to_utf8_bytes(f["content"])
        tree = parser.parse(code_bytes)
        root = tree.root_node
        
        file_struct = {
            "functions": [],
            "classes": []
        }

        spaces[f["path"]] = walk(
            root,
            f,
            code_bytes,
            function_index,
            raw_call_graph,
            file_struct
        )

        dir_structure[f["path"]] = file_struct
        
    # Resolve Call Graph
    logging.info("Resolving call graph")

    call_graph = {}

    for caller, callees in raw_call_graph.items():
        call_graph[caller] = list()

        for callee in callees:
            if callee in function_index:
                call_graph[caller].append(function_index[callee])
            else:
                call_graph[caller].append(callee)  # unresolved / external
    
    logging.info("Persisting call graph and directory structure")

    json.dump(call_graph, open('database/call_graph.json', 'w'), indent=2)
    json.dump(dir_structure, open('database/dir_structure.json', 'w'), indent=2)
    
    def delete_after_exit(path):
        logging.info(f"Scheduling cleanup for directory: {path}")

        subprocess.Popen(
            [
                "cmd", "/c",
                f"timeout /t 1 >nul && rmdir /s /q {path} && mkdir {path}"
            ],
            creationflags=subprocess.CREATE_NO_WINDOW
        )

    delete_after_exit("public")
    logging.info("Code extraction completed successfully")

    return spaces

if __name__ == "__main__":
    spaces = extract_spaces(language="python")