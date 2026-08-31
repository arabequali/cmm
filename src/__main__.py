"""Entry point for the call me maybe function calling tool."""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError


class ParameterDef(BaseModel):
    """Definition of a single function parameter."""

    type: str


class FunctionDef(BaseModel):
    """Definition of a callable function."""

    name: str
    description: str
    parameters: dict[str, ParameterDef]
    returns: ParameterDef


class Prompt(BaseModel):
    """A single natural language prompt."""

    prompt: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to the output JSON file.",
    )
    return parser.parse_args()


def load_json_file(path: str) -> list[dict]:
    """Load and parse a JSON file safely.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed list of dicts from the file.
    """
    file_path = Path(path)

    if not file_path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(f"Error: expected a JSON array in {path}", file=sys.stderr)
        sys.exit(1)

    return data


def load_functions(path: str) -> list[FunctionDef]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        List of validated FunctionDef objects.
    """
    raw = load_json_file(path)
    functions = []
    for item in raw:
        try:
            functions.append(FunctionDef(**item))
        except (ValidationError, TypeError) as e:
            print(f"Warning: skipping invalid function definition: {e}",
                  file=sys.stderr)
    return functions


def load_prompts(path: str) -> list[Prompt]:
    """Load and validate prompts from a JSON file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        List of validated Prompt objects.
    """
    raw = load_json_file(path)
    prompts = []
    for item in raw:
        try:
            prompts.append(Prompt(**item))
        except (ValidationError, TypeError) as e:
            print(f"Warning: skipping invalid prompt: {e}", file=sys.stderr)
    return prompts


def main() -> None:
    """Main entry point for the function calling tool."""
    args = parse_args()

    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    print(f"Loaded {len(functions)} function(s) and {len(prompts)} prompt(s).")

    from vocab import load_model, load_vocabulary, build_reverse_vocab
    from pipeline import run_pipeline

    model = load_model()
    vocab = load_vocabulary(model)
    build_reverse_vocab(vocab)

    print(f"Vocabulary loaded: {len(vocab)} tokens.")

    results = run_pipeline(model, vocab, prompts, functions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")
    except OSError as e:
        print(f"Error: could not write output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
