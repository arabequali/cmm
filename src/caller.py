def build_function_selection_prompt(
    prompt: str,
    functions: list[FunctionDef]
) -> str:
    fn_list = "\n".join(
        f"- {f.name}: {f.description}" for f in functions
    )
    return (
        f"Available functions:\n{fn_list}\n\n"
        f"User request: \"{prompt}\"\n\n"
        f"Function to call: "
    )

def build_argument_prompt(
    prompt: str,
    fn: FunctionDef,
    param_name: str,
    already_filled: dict
) -> str:
    return (
        f"User request: \"{prompt}\"\n"
        f"Calling function: {fn.name}\n"
        f"Already determined: {already_filled}\n"
        f"What is the value of parameter '{param_name}'? "
    )
