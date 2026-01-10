# utils/helper.py

# Formatting the retrieved elements into a string context
def formatContext(retrieved_elements):
    return "\n".join(doc.metadata["text"] for doc in retrieved_elements['matches'])

# Ensure text is in UTF-8 bytes
def to_utf8_bytes(text):
    if isinstance(text, bytes):
        return text
    return text.encode("utf-8", errors="replace")

