import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    # q , k transpose : dot 
    # root k dim 
    # softmax (QKt . V ) understand 
    dim_k = torch.sqrt(torch.tensor(K.shape[-1])) 
    inner_dot = torch.matmul(Q, K.mT)
    partA = inner_dot / dim_k
    prob_partA = torch.softmax(partA, dim=-1)
    attention_score = torch.matmul(prob_partA, V)
    return attention_score
    