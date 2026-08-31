"""Main pipeline: prompt -> function call."""

import sys
from typing import Any

from llm_sdk import Small_LLM_Model

from __main__ import FunctionDef, Prompt
from decoder import select_function, extract_argument


def build_prompt(
    prompt: str,
    functions: list[FunctionDef],
) -> str:
    """Build the text prompt sent to the LLM.

    Args:
        prompt: The natural language user request.
        functions: Available function definitions.

    Returns:
        Formatted prompt string.
    """
    fn_descriptions = "\n".join(
        f"- {fn.name}: {fn.description}" for fn in functions
    )
    return (
        f"Available functions:\n{fn_descriptions}\n\n"
        f"User request: {prompt}\n"
        f"Function to call: "
    )


def run_pipeline(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompts: list[Prompt],
    functions: list[FunctionDef],
) -> list[dict[str, Any]]:
    """Run the full function calling pipeline for all prompts.

    Args:
        model: The loaded LLM model.
        vocab: token_id -> token_string mapping.
        prompts: List of user prompts to process.
        functions: Available function definitions.

    Returns:
        List of result dicts ready to serialize as JSON.
    """
    fn_map = {fn.name: fn for fn in functions}
    fn_names = list(fn_map.keys())
    results: list[dict[str, Any]] = []

    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i + 1}/{len(prompts)}: {prompt.prompt}")
        try:
            # 1. Tokenize the prompt
            text = build_prompt(prompt.prompt, functions)
            input_ids: list[int] = model.encode(text).squeeze().tolist()

            # 2. Select the function
            fn_name, input_ids = select_function(
                model, input_ids, vocab, fn_names
            )

            if fn_name not in fn_map:
                print(f"  Warning: model chose unknown function '{fn_name}'",
                      file=sys.stderr)
                continue

            fn_def = fn_map[fn_name]

            # 3. Extract each argument
            parameters: dict[str, Any] = {}
            for param_name, param_def in fn_def.parameters.items():
                context = f", {param_name}="
                extra_ids: list[int] = (
                    model.encode(context).squeeze().tolist()
                )
                input_ids = input_ids + extra_ids

                value, input_ids = extract_argument(
                    model, input_ids, vocab, param_def.type
                )
                parameters[param_name] = value

            results.append({
                "prompt": prompt.prompt,
                "name": fn_name,
                "parameters": parameters,
            })
            print(f"  -> {fn_name}({parameters})")

        except Exception as e:
            print(f"  Error processing prompt: {e}", file=sys.stderr)

    return results
