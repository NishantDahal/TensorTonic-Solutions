import torch
import torch.nn as nn

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Returns an embedding layer with the requested dimensions.
    """
    embedding = nn.Embedding(vocab_size, d_model)
    return embedding

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Returns scaled token embeddings.
    """
    d_model = torch.tensor(d_model)
    emb = embedding(tokens)
    scaled_token_embeddings = emb * torch.sqrt(d_model)
    return scaled_token_embeddings