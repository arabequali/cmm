*This project has been created as part of the 42 curriculum by avally.*

# call me maybe — Introduction to Function Calling in LLMs

## Description

This project implements a **function calling tool** that translates natural language
prompts into structured, machine-executable function calls using a small LLM
(Qwen/Qwen3-0.6B, ~600M parameters).

Given a prompt like "What is the sum of 2 and 3?", the tool does not answer "5".
Instead it produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The key challenge is reliability: small LLMs produce valid JSON only ~30% of the
time when simply prompted. This project achieves near-100% valid output through
**constrained decoding** — guiding the model's token selection at every generation
step rather than relying on prompting alone.

---

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone <your_repo_url>
cd call-me-maybe
uv sync
```

### Running the program

With default paths:
```bash
make run
```

With custom paths:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Other commands

```bash
make install      # install dependencies
make debug        # run with Python debugger
make lint         # flake8 + mypy checks
make lint-strict  # mypy --strict mode
make clean        # remove __pycache__, .mypy_cache
```

---

## Algorithm Explanation

The constrained decoding pipeline runs in three phases for each prompt:

### Phase 1 — Function selection

A prompt is built from the natural language request and the list of available
function names and descriptions. The LLM is then run token by token, but at
each step only tokens that are valid continuations of one of the known function
names are allowed. All other tokens are masked to `-inf` in the logits before
argmax selection. This guarantees the output is always a real function name.

### Phase 2 — Argument extraction

Once the function is chosen, its parameter schema is known. For each parameter,
the model generates a value under type-specific constraints:

- **number**: only tokens composed of digits `0-9` and `.` are allowed,
  with a maximum length of 10 characters to prevent runaway generation
- **string**: all tokens allowed between opening and closing `"` markers,
  filtering out newlines, BPE special characters, and quote-containing tokens
- **boolean**: constrained to exactly `true` or `false`

### Phase 3 — Output assembly

Results are collected and serialized to a JSON array in
`data/output/function_calls.json`.

---

## Design Decisions

**Vocabulary-based masking** — the vocabulary JSON maps every token ID to its
string. At each generation step we scan this map to find which token IDs are
consistent with the current partial output, then mask the rest to `-inf`.
This is the core of constrained decoding and guarantees structural validity.

**Pydantic for all data models** — input validation is strict and failures are
reported per-entry rather than crashing the whole run.

**Per-type generators** — each JSON type (`number`, `string`, `boolean`) has
its own generation function with tailored constraints, keeping the logic simple
and easy to extend.

**Token length caps** — string arguments use a max token limit (50 for the
first parameter, 15 for subsequent ones) to prevent the model from
over-generating in regex and replacement fields.

**Graceful error handling** — every file read, model call, and generation step
is wrapped in try/except. A single bad prompt never kills the whole batch.

---

## Performance Analysis

| Metric | Target | Achieved |
|---|---|---|
| Valid JSON output | 100% | 100% |
| Correct function selection | 90%+ | ~90% |
| Processing speed | < 5 min | ~1-2 min on CPU |

The Qwen3-0.6B model runs on CPU in WSL2. Constrained decoding adds minimal
overhead per token since the vocabulary scan is O(vocab_size) with a fixed
vocab of 151,643 tokens.

---

## Challenges Faced

**Token boundary misalignment** — tokens rarely align with word boundaries.
A function name like `fn_add_numbers` may be split across several tokens
unpredictably. The prefix-matching approach handles this by allowing any token
whose string is a prefix of the target or vice versa.

**BPE special characters** — the tokenizer uses `Ġ` for spaces and `Ċ` for
newlines internally. String extraction had to explicitly filter these out to
avoid garbage in the output values.

**Number over-generation** — without a length cap, the model would generate
arbitrarily long decimals like `3.000000000000001`. Fixed by limiting number
generation to 10 characters and only allowing digit and dot tokens.

**String over-generation** — for regex parameters, the model would keep
generating patterns far beyond what was needed. Fixed by using shorter token
limits for non-first string parameters.

**CUDA vs CPU** — PyTorch defaults to the CUDA version which requires NVIDIA
drivers not available in WSL2. Solved by installing the CPU-only build of
PyTorch explicitly.

---

## Testing Strategy

1. **Manual spot checks** — run with the provided example files and verify
   output matches expected function names and argument values.

2. **Edge cases tested**:
   - Simple arithmetic (sum of 2 and 3, sum of 265 and 345)
   - String arguments (greet john, greet shrek)
   - String reversal (hello, world)
   - Square root extraction
   - Regex substitution with multiple parameters

3. **Error handling tested**:
   - Missing input files
   - Malformed JSON in input files
   - Unknown function names generated by the model

4. **JSON validation** — the output file is re-parsed after writing to confirm
   it is well-formed.

---

## Example Usage

```bash
# Basic run with default input files
make run

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/results.json
```

Expected terminal output:
```
Loaded 5 function(s) and 11 prompt(s).
Vocabulary loaded: 151643 tokens.
Processing prompt 1/11: What is the sum of 2 and 3?
-> fn_add_numbers({'a': 2.0, 'b': 3.0})
Processing prompt 2/11: What is the sum of 265 and 345?
-> fn_add_numbers({'a': 265.0, 'b': 345.0})
...
Results written to data/output/function_calls.json
```

---

## Resources

### Documentation and articles

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic v2 documentation](https://docs.pydantic.dev/)
- [BPE tokenization explained](https://huggingface.co/learn/nlp-course/chapter6/5)
- [JSON Schema specification](https://json-schema.org/)
- [uv package manager](https://github.com/astral-sh/uv)

### AI usage

AI (Claude) was used for:
- Scaffolding the initial project structure and boilerplate
- Drafting docstrings and type hint patterns

All generated code was reviewed, tested, and adapted. No AI-generated code
was submitted without full understanding of its behaviour.
