from transformers import AutoTokenizer, AutoModel
import torch
from langchain.embeddings.base import Embeddings

# Load CodeBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Function to generate embeddings using CodeBERT
def embed_text(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# Custom Embeddings Class
class CodeBERTEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [embed_text(text) for text in texts]

    def embed_query(self, text):
        return embed_text(text)

EMBEDDING_DIM = 768