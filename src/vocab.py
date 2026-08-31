"""Vocabulary loading and token mapping utilities."""

import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model


def load_model(model_name: str = "Qwen/Qwen3-0.6B") -> Small_LLM_Model:
    """Load the LLM model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded Small_LLM_Model instance.
    """
    try:
        model = Small_LLM_Model(model_name)
        return model
    except Exception as e:
        print(f"Error: could not load model '{model_name}': {e}",
              file=sys.stderr)
        sys.exit(1)


def load_vocabulary(model: Small_LLM_Model) -> dict[int, str]:
    """Load the token vocabulary from the model's vocab JSON file.

    Args:
        model: The loaded LLM model instance.

    Returns:
        Dictionary mapping token_id (int) -> token_string (str).
    """
    vocab_path = model.get_path_to_vocab_file()

    try:
        with open(Path(vocab_path), "r", encoding="utf-8") as f:
            raw: dict[str, int] = json.load(f)
    except FileNotFoundError:
        print(f"Error: vocabulary file not found at {vocab_path}",
              file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in vocabulary file: {e}",
              file=sys.stderr)
        sys.exit(1)

    # vocab.json maps token_string -> token_id, we want the reverse
    vocab: dict[int, str] = {v: k for k, v in raw.items()}
    return vocab


def build_reverse_vocab(vocab: dict[int, str]) -> dict[str, int]:
    """Build a reverse mapping from token string to token ID.

    Args:
        vocab: Forward mapping token_id -> token_string.

    Returns:
        Reverse mapping token_string -> token_id.
    """
    return {v: k for k, v in vocab.items()}


def get_token_ids_with_prefix(
    prefix: str,
    vocab: dict[int, str]
) -> list[int]:
    """Find all token IDs consistent with a given prefix.

    A token is valid if the current prefix could continue into it,
    or if it could be the start of the prefix.

    Args:
        prefix: The string built so far that we want to continue.
        vocab: Forward mapping token_id -> token_string.

    Returns:
        List of valid token IDs consistent with the prefix.
    """
    valid: list[int] = []
    for token_id, token_str in vocab.items():
        if token_str.startswith(prefix) or prefix.startswith(token_str):
            valid.append(token_id)
    return valid
