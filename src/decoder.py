"""Constrained decoding engine for guaranteed valid JSON function calls."""

import sys
from typing import Any

from llm_sdk import Small_LLM_Model


def get_next_token_constrained(
    model: Small_LLM_Model,
    input_ids: list[int],
    valid_token_ids: list[int],
) -> int:
    """Run one forward pass and pick the best token among valid ones.

    Args:
        model: The loaded LLM model.
        input_ids: Current sequence of token IDs.
        valid_token_ids: Token IDs allowed at this step.

    Returns:
        The chosen next token ID.
    """
    logits = model.get_logits_from_input_ids(input_ids)

    # Set all invalid tokens to -inf
    masked: list[float] = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        if tid < len(logits):
            masked[tid] = logits[tid]

    return int(masked.index(max(masked)))


def generate_exact_string(
    model: Small_LLM_Model,
    input_ids: list[int],
    vocab: dict[int, str],
    allowed_strings: list[str],
) -> tuple[str, list[int]]:
    """Generate exactly one of the allowed strings using constrained decoding.

    Args:
        model: The loaded LLM model.
        input_ids: Current token ID sequence.
        vocab: token_id -> token_string mapping.
        allowed_strings: The only strings the model may produce.

    Returns:
        Tuple of (chosen string, updated input_ids).
    """
    generated_text = ""
    current_ids = input_ids[:]

    for _ in range(200):
        valid_ids: list[int] = []
        for token_id, token_str in vocab.items():
            candidate = generated_text + token_str
            for allowed in allowed_strings:
                if allowed.startswith(candidate) or candidate == allowed:
                    valid_ids.append(token_id)
                    break

        if not valid_ids:
            break

        next_token = get_next_token_constrained(model, current_ids, valid_ids)
        token_str = vocab[next_token]
        generated_text += token_str
        current_ids.append(next_token)

        if generated_text in allowed_strings:
            break

    return generated_text, current_ids


def generate_number(
    model: Small_LLM_Model,
    input_ids: list[int],
    vocab: dict[int, str],
) -> tuple[float, list[int]]:
    """Generate a valid number using constrained decoding.

    Args:
        model: The loaded LLM model.
        input_ids: Current token ID sequence.
        vocab: token_id -> token_string mapping.

    Returns:
        Tuple of (parsed float value, updated input_ids).
    """
    NUMBER_CHARS = set("0123456789")
    DECIMAL_CHARS = set("0123456789.")
    generated = ""
    current_ids = input_ids[:]

    for _ in range(20):
        # Only allow digit tokens, and dot only if no dot yet
        valid_ids = [
            tid for tid, tok in vocab.items()
            if tok
            and all(c in DECIMAL_CHARS for c in tok)
            and not (tok == "." and "." in generated)
            and not (tok == "." and generated == "")
        ]

        if not valid_ids:
            break

        next_token = get_next_token_constrained(
            model, current_ids, valid_ids
        )
        token_str = vocab[next_token]

        # Stop if adding this token makes no sense as a number
        candidate = generated + token_str
        try:
            float(candidate)
        except ValueError:
            break

        generated += token_str
        current_ids.append(next_token)

        # Stop after a reasonable number length
        if len(generated) >= 30:
            break

    try:
        return float(generated), current_ids
    except ValueError:
        return 0.0, current_ids


def generate_string_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    vocab: dict[int, str],
) -> tuple[str, list[int]]:
    """Generate a string value stopping at the closing quote token.

    Args:
        model: The loaded LLM model.
        input_ids: Current token ID sequence.
        vocab: token_id -> token_string mapping.

    Returns:
        Tuple of (string content without quotes, updated input_ids).
    """
    generated = ""
    current_ids = input_ids[:]

    # All token IDs that are exactly a closing quote
    close_quote_ids = set(
        tid for tid, tok in vocab.items() if tok == '"'
    )

    # Valid content tokens: no quotes, no newlines, no BPE special chars
    content_token_ids = [
        tid for tid, tok in vocab.items()
        if '"' not in tok
        and "\n" not in tok
        and "\r" not in tok
        and "Ċ" not in tok
        and "Ġ," not in tok   # stop before ", param=" patterns
    ]

    for _ in range(50):
        # Only allow content tokens OR a closing quote
        valid_ids = content_token_ids + list(close_quote_ids)
        next_token = get_next_token_constrained(
            model, current_ids, valid_ids
        )

        # Stop immediately on closing quote
        if next_token in close_quote_ids:
            current_ids.append(next_token)
            break

        token_str = vocab[next_token]

        # Extra safety: stop if token contains a quote
        if '"' in token_str:
            break

        generated += token_str
        current_ids.append(next_token)

    # Clean up any BPE space markers from the output
    cleaned = generated.replace("Ġ", " ").strip()
    return cleaned, current_ids


def generate_boolean(
    model: Small_LLM_Model,
    input_ids: list[int],
    vocab: dict[int, str],
) -> tuple[bool, list[int]]:
    """Generate a boolean value (true or false).

    Args:
        model: The loaded LLM model.
        input_ids: Current token ID sequence.
        vocab: token_id -> token_string mapping.

    Returns:
        Tuple of (bool value, updated input_ids).
    """
    result, new_ids = generate_exact_string(
        model, input_ids, vocab, ["true", "false"]
    )
    return result == "true", new_ids


def select_function(
    model: Small_LLM_Model,
    input_ids: list[int],
    vocab: dict[int, str],
    function_names: list[str],
) -> tuple[str, list[int]]:
    """Use the LLM to select which function to call.

    Args:
        model: The loaded LLM model.
        input_ids: Tokenized prompt.
        vocab: token_id -> token_string mapping.
        function_names: List of valid function name strings.

    Returns:
        Tuple of (chosen function name, updated input_ids).
    """
    return generate_exact_string(model, input_ids, vocab, function_names)


def extract_argument(
    model: Small_LLM_Model,
    input_ids: list[int],
    vocab: dict[int, str],
    param_type: str,
) -> tuple[Any, list[int]]:
    """Extract one argument of the given type using constrained decoding.

    Args:
        model: The loaded LLM model.
        input_ids: Current token ID sequence.
        vocab: token_id -> token_string mapping.
        param_type: One of 'number', 'string', 'boolean'.

    Returns:
        Tuple of (extracted value, updated input_ids).
    """
    if param_type == "number":
        return generate_number(model, input_ids, vocab)
    elif param_type == "string":
        open_quote_ids = [tid for tid, tok in vocab.items() if tok == '"']
        if open_quote_ids:
            next_token = get_next_token_constrained(
                model, input_ids, open_quote_ids
            )
            input_ids = input_ids + [next_token]
        return generate_string_value(model, input_ids, vocab)
    elif param_type == "boolean":
        return generate_boolean(model, input_ids, vocab)
    else:
        return generate_string_value(model, input_ids, vocab)
