import json
from typing import List
from llm_sdk import Small_LLM_Model
from src.models import FunctionDef
from src.vocab import load_vocab
from src.generator import generate_next_token

def get_tokens_matching_prefix(
    vocab: dict[int, str],
    prefix: str
) -> List[int]:
    """Return token IDs whose string starts with the given prefix."""
    return [
        tid for tid, tok in vocab.items()
        if tok.startswith(prefix) or prefix.startswith(tok)
    ]

def generate_constrained_string(
    model: Small_LLM_Model,
    input_ids: List[int],
    vocab: dict[int, str],
    allowed_strings: List[str]
) -> tuple[str, List[int]]:
    """
    Generate one of the allowed_strings token-by-token.
    Returns the chosen string and updated input_ids.
    """
    current = ""
    ids = list(input_ids)
    
    while True:
        candidates = [s for s in allowed_strings if s.startswith(current)]
        if not candidates:
            break
        if len(candidates) == 1 and candidates[0] == current:
            break
        
        valid_tids = []
        for tid, tok in vocab.items():
            for candidate in candidates:
                if candidate.startswith(current + tok):
                    valid_tids.append(tid)
                    break
        
        if not valid_tids:
            break
            
        next_tid = generate_next_token(model, ids, valid_tids)
        next_tok = vocab[next_tid]
        current += next_tok
        ids.append(next_tid)
    
    return current, ids
