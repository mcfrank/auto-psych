# Game of 24: Think-Aloud Study — Problem Definition

## Task

**Game of 24**: Given 4 numbers, find an arithmetic expression using +, -, ×, ÷ (each number used exactly once) that equals the target value. In this dataset the target is always 24. Participants verbalized their thinking aloud while solving each puzzle, and their think-aloud speech was transcribed and coded.

The task is a combinatorial search problem: at each step the solver selects two numbers from the available set, applies an arithmetic operation, and replaces the pair with the result. This continues until one number remains. If it equals the target, the puzzle is solved.

## Dataset

- **Participants**: 640
- **Trials per participant**: ~20
- **Data file**: `/Users/ben/Documents/llm-verbal-protocol/data/coded/full-experiment/full-experiment_model-claude-3-5-sonnet-20241022.csv`

Each row corresponds to one participant solving one puzzle. The key columns are:

| Column | Description |
|--------|-------------|
| `choices` | JSON list string of the 4 available numbers, e.g. `"[5, 5, 12, 12]"` |
| `target` | Target number as a string, e.g. `"24.0"` |
| `correct` | 1 if the participant submitted a correct answer, 0 otherwise |
| `lm_code_translation` | Python code reconstructing the participant's search trace using the `GraphBuilder` API (see below) |

## Stimulus format

- **`sequence_a`** (pipeline name; raw column is `choices`): JSON list string of the 4 available numbers, e.g. `"[5, 5, 12, 12]"`. Parse with `import json; choices = json.loads(stimulus[0])`.
- **`sequence_b`** (pipeline name; raw column is `target`): Target number as a string, e.g. `"24.0"`. Parse with `target = int(float(stimulus[1]))`.

## Response format

- **`chose_left`** (pipeline name; raw column is `correct`): 1 if the participant submitted a correct answer, 0 otherwise. The pipeline treats `chose_left == 1` as "left" and `chose_left == 0` as "right", so model predictions should return `{"left": p_solved, "right": 1 - p_solved}`.

## Extra column: `lm_code_translation`

Each row includes `lm_code_translation`, a Python code string that reconstructs the participant's search trace using the `GraphBuilder` API. In aggregated test-statistic rows, this appears as `lm_code_translation_list` — a list of code strings for all participants who attempted that stimulus.

### GraphBuilder API

The `GraphBuilder` class (available at `/Users/ben/Documents/llm-verbal-protocol/src/preproc/reasoning_graph.py`) provides a structured representation of a participant's search process:

```python
graph = GraphBuilder(start_state)
# start_state is a sorted tuple of ints, e.g. (5, 5, 12, 12)

graph.explore_operation(
    curr_state,
    operation="12+5=17",      # string describing the arithmetic step
    resulting_state=(5, 17),  # resulting numbers after the operation
    comment="..."             # optional comment
)

graph.move_to_node(state)
# backtrack or jump to a previously visited state

graph.set_subgoal(subgoal_state, ...)
# set an intermediate target the participant is working toward
```

Each `explore_operation` call represents one arithmetic step the participant considered (they may later backtrack). `move_to_node` calls represent backtracking or jumping to a previously visited state. `set_subgoal` calls indicate the participant explicitly named an intermediate target.

### Parsing `lm_code_translation`

To count operations in a trace, count occurrences of `explore_operation` calls. To detect backtracking, count `move_to_node` calls. Do not execute the code — parse it as a string.

## Cognitive model format

Each cognitive model is a Python function:

```python
def model_name(stimulus, response_options):
    """Brief docstring."""
    import json, random, itertools
    choices = json.loads(stimulus[0])          # list of 4 ints
    target = int(float(stimulus[1]))           # int (24 in this dataset)
    # response_options == ["left", "right"]
    # "left" = correct (solved), "right" = incorrect (failed)

    N = 50  # Monte Carlo simulations
    solved = 0
    for _ in range(N):
        if simulate_once(choices, target):
            solved += 1
    p_solved = solved / N
    return {"left": p_solved, "right": 1.0 - p_solved}
```

### Key implementation requirements

- **Self-contained**: use only Python stdlib modules: `math`, `random`, `itertools`, `json`, `collections`. Do NOT import from `llm-verbal-protocol` or any non-stdlib module.
- **Monte Carlo**: run N=50 simulations per call. Return `{"left": fraction_solved, "right": 1 - fraction_solved}`.
- **Stochastic search algorithms**: models should implement DFS, BFS, best-first search, random walk, or similar. The stochasticity determines when a solver finds the solution.

### Generating successors (implement inline)

To enumerate all states reachable from a state (list of ints), try all pairs and all four operations:

```python
def generate_successors(nums):
    """Return list of (result_nums, op_str) for all valid one-step operations."""
    import itertools
    successors = []
    for i, j in itertools.combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
        for result, op_str in [
            (a + b, f"{a}+{b}={a+b}"),
            (a * b, f"{a}*{b}={a*b}"),
            (b - a, f"{b}-{a}={b-a}"),
            (a - b, f"{a}-{b}={a-b}"),
            (b / a if a != 0 else None, f"{b}/{a}={b/a:.0f}" if a != 0 else None),
            (a / b if b != 0 else None, f"{a}/{b}={a/b:.0f}" if b != 0 else None),
        ]:
            if result is not None and result == int(result) and result >= 0:
                successors.append((sorted(rest + [int(result)]), op_str))
    return successors
```

### Reference implementations (for inspiration — do NOT import)

- `/Users/ben/Documents/llm-verbal-protocol/src/models/countdown_dfs.py` — stochastic DFS with softmax node selection weighted by heuristic
- `/Users/ben/Documents/llm-verbal-protocol/src/models/countdown_bestfirst.py` — best-first search variant
- `/Users/ben/Documents/llm-verbal-protocol/src/models/countdown_utils.py` — `generate_successors`, `sum_heuristic`, `combo_heuristic`, and other utilities

These use numpy/torch; your models must reproduce equivalent logic using only stdlib.

## Scientific goal

Find computational models that predict **when humans succeed vs. fail** at Game of 24, and reveal **what aspects of human search behavior** differ from pure computational models. Key questions:

- Do humans behave more like DFS, BFS, or best-first search?
- Which heuristics (sum-to-target, factor-proximity, etc.) best capture human difficulty judgments?
- Does the search depth reflected in participants' `lm_code_translation` traces correlate with model-predicted difficulty?
- Are there systematic biases (e.g., preference for multiplication, avoidance of fractions) that improve model fit?
