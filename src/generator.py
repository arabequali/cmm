import numpy as np
from typing import List
from llm_sdk import Small_LLM_Model

def softmax(logits: List[float]) -> np.ndarray:
    arr = np.array(logits)
    arr -= arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()

def generate_next_token(
    model: Small_LLM_Model,
    input_ids: List[int],
    valid_token_ids: List[int]
) -> int:
    """Get logits, mask invalid tokens, return best valid token."""
    logits = model.get_logits_from_input_ids(input_ids)
    
    masked = np.full(len(logits), -np.inf)
    
    for tid in valid_token_ids:
        if tid < len(logits):
            masked[tid] = logits[tid]
    
    return int(np.argmax(masked))
