# DSPy Prompt Optimizer

A Streamlit app that lets you upload any tabular dataset, define input features and a target column, run every DSPy optimizer against it, compare scores side-by-side, and export the final optimized prompt as JSON — all without writing a single line of code.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![DSPy](https://img.shields.io/badge/dspy-3.2.1-orange)
![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-red)

---

## What it does

| Tab | Purpose |
|-----|---------|
| **Dataset** | Upload a training CSV and an optional held-out test CSV. Six built-in topic examples let you try the app instantly. |
| **Variables** | Pick input (X) columns, the target (Y) column, write a task description, and choose an evaluation metric. |
| **Optimization** | Select one optimizer or compare all seven at once. Set dataset size guardrails, review LLM-call cost estimates, then click Run. |
| **Results** | Baseline vs. optimized scores in a comparison table and bar chart. Per-run inspector with instructions, few-shot examples, and a JSON download. Live inference section to test the compiled program immediately. |

### Supported optimizers

| Optimizer | Strategy | Needs val set? |
|-----------|----------|---------------|
| **LabeledFewShot** | Selects k labeled examples as demos — fast sanity-check baseline | No |
| **BootstrapFewShot** | Teacher–student trace bootstrapping | No |
| **BootstrapFewShotWithRandomSearch** | Bootstrap + random search over candidate programs | Yes |
| **BootstrapFewShotWithOptuna** | Bootstrap + Optuna Bayesian hyperparameter search | Yes |
| **COPRO** | Coordinate-ascent instruction optimization (no demos) | No |
| **MIPROv2** | Bayesian joint optimization of instructions + demos | Yes |
| **GEPA** *(experimental)* | Evolutionary optimizer with LM-driven reflection | Yes |

### Evaluation metrics

- **Exact Match** — case-insensitive, whitespace-normalized string equality
- **Contains Answer** — checks whether the prediction contains the gold answer
- **F1 Token Overlap** — word-level F1 (good for extractive QA)
- **Always True** — scores every prediction 1.0 (useful for generation tasks)

---

## Prerequisites

- Python 3.10 or 3.11 (DSPy 3.x does **not** support Python 3.12 yet)
- An LLM API key — OpenAI, Anthropic, or any provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers)
- `conda` (recommended) or a standard `venv`

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Laxminarayen/dspy-prompt-optimizer.git
cd dspy-prompt-optimizer
```

### 2. Create a Python environment

**With conda (recommended):**

```bash
conda create -n dspy python=3.10 -y
conda activate dspy
```

**With venv:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins:

```
streamlit>=1.35.0
dspy>=3.0.0
pandas>=2.0.0
optuna>=3.0.0
```

> **Note:** DSPy 3.x pulls in LiteLLM, so nearly every major LLM provider works out of the box with no extra packages.

### 4. Run the app

**With conda:**

```bash
conda run -n dspy python -m streamlit run app.py
```

**With venv (environment already active):**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage walkthrough

### Step 1 — Configure LLM (sidebar)

Enter your provider, model name, and API key. Examples:

| Provider | Model string | Key env var |
|----------|-------------|-------------|
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| Groq | `groq/llama-3.1-8b-instant` | `GROQ_API_KEY` |
| Ollama (local) | `ollama/llama3` | *(no key needed)* |

The model string follows LiteLLM's `provider/model` format. Cheaper/faster models work well for exploration; stronger models give better optimization results.

### Step 2 — Upload a dataset (Tab 1)

Upload a CSV with at least two columns — one for input(s) and one for the answer/label. Optionally upload a separate test CSV to get honest held-out evaluation scores.

If you don't have a dataset handy, click **Load example** to load one of six built-in topic examples.

> For SQuAD-style QA testing, download the validation split from HuggingFace:
> ```python
> from datasets import load_dataset
> ds = load_dataset("rajpurkar/squad", split="validation")
> ds.to_pandas().rename(columns={"answers": "answer"}) \
>   .assign(answer=lambda df: df["answer"].map(lambda a: a["text"][0])) \
>   [["title","context","question","answer"]] \
>   .to_csv("squad_validation.csv", index=False)
> ```

### Step 3 — Define variables (Tab 2)

- **X (inputs):** multi-select the columns the LLM should receive as input
- **Target (Y):** the column containing ground-truth answers
- **Task description:** one sentence describing what the LLM should do (used in instruction proposals)
- **Metric:** choose the evaluation metric that fits your task

### Step 4 — Run optimization (Tab 3)

**Single mode** — pick one optimizer, tune its parameters, run it.

**Compare mode** — tick multiple optimizers (or "Select all"), review the cost estimate table, then run all at once.

Size guardrails cap how many rows are sent to the optimizer and evaluator so you don't accidentally burn through API credits on a large dataset.

### Step 5 — Review results (Tab 4)

- Baseline score (zero-shot, no optimization) is shown as a reference
- Comparison table with Δ Baseline, demos added, instruction changed flag, and timing
- Bar chart for quick visual comparison
- Per-run inspector: expand any run to see the full instructions, few-shot examples, and input/output field names used by each predictor
- **Download prompt JSON** — save the compiled program definition for use in your own code
- **Live inference** — type an input directly in the UI and call the compiled program against the LLM in real time

---

## Using the exported JSON in your own code

```python
import dspy, json

# Load the JSON produced by the Download button
with open("my_optimized_prompt.json") as f:
    data = json.load(f)

# Reconstruct the module
lm = dspy.LM(model="openai/gpt-4o-mini", api_key="sk-...")
with dspy.context(lm=lm):
    predictor = dspy.Predict(data["signature"])
    # Inject the optimized instructions back
    for name, pred_data in data["predictors"].items():
        predictor.signature = predictor.signature.with_instructions(
            pred_data["instructions"]
        )
    result = predictor(**your_inputs)
    print(result)
```

---

## Project structure

```
dspy-prompt-optimizer/
├── app.py            # Single-file Streamlit application
├── requirements.txt  # Python dependencies
└── README.md
```

Everything lives in `app.py`. Key sections:

| Lines (approx.) | Section |
|----------------|---------|
| 1–260 | Top-level helper functions (`build_result_json`, `create_metric_fn`, `dispatch_optimizer`, `evaluate_compiled`, `record_run`, `_estimate_llm_calls`, `_df_uploader`) |
| 261–320 | Streamlit page config, sidebar LLM configuration |
| 320–560 | Tab 1 (Dataset), Tab 2 (Variables) |
| 561–960 | Tab 3 (Optimization) — OPTIMIZERS catalog, mode selector, guardrails, run loop |
| 961–end | Tab 4 (Results) — comparison table, chart, per-run inspector, live inference |

---

## Contributing

Contributions of all sizes are welcome — from fixing a typo in a tooltip to adding a full RAG pipeline. This section covers everything you need to go from zero to a merged pull request.

---

### 1. Fork and clone

```bash
# 1. Click "Fork" on the GitHub repo page, then:
git clone https://github.com/<your-username>/dspy-prompt-optimizer.git
cd dspy-prompt-optimizer

# 2. Add the upstream remote so you can pull future changes
git remote add upstream https://github.com/Laxminarayen/dspy-prompt-optimizer.git
```

---

### 2. Set up your development environment

```bash
# Create an isolated conda environment (Python 3.10 or 3.11 only — DSPy 3.x does not support 3.12)
conda create -n dspy-dev python=3.10 -y
conda activate dspy-dev

# Install all dependencies
pip install -r requirements.txt
```

If you are adding a feature that requires a new package (e.g., `rouge-score` for a new metric), install it and add it to `requirements.txt` with a `>=` version pin:

```bash
pip install rouge-score
echo "rouge-score>=0.1.2" >> requirements.txt
```

---

### 3. Run the app locally

```bash
# With conda env active:
conda run -n dspy-dev python -m streamlit run app.py

# Or if the env is already activated:
streamlit run app.py
```

The app opens at `http://localhost:8501`. Streamlit hot-reloads on every file save, so you can edit `app.py` and see changes instantly without restarting.

---

### 4. Understand the codebase before you change it

Everything lives in a single file — `app.py`. Read through these sections before making changes:

| What to read first | Why |
|--------------------|-----|
| `create_metric_fn()` (line ~86) | Understand how metrics work before adding a new one |
| `make_dspy_module()` (line ~125) | Understand how DSPy signatures are built from column names |
| `dispatch_optimizer()` (line ~139) | Understand how each optimizer is called — add new ones here |
| `OPTIMIZERS` dict (line ~567) | Catalog that drives the UI — every optimizer's params are declared here |
| `_to_examples()` (inside the run button, line ~896) | How DataFrame rows become `dspy.Example` objects |
| `record_run()` (line ~234) | How results are stored in session state for the Results tab |

Key design constraints:
- **All DSPy calls must be inside `with dspy.context(lm=lm):`** — `dspy.configure()` is not thread-safe in DSPy 3.x and will raise a `RuntimeError` in Streamlit.
- **Column names are sanitized** to valid Python identifiers before building signatures. If you add a new place that uses column names as DSPy field names, apply the same `re.sub(r"[^a-zA-Z0-9_]", "_", name)` pattern.
- **`dispatch_optimizer()` receives a copy of params** (`dict(params_per_optimizer[opt_name])`). Always `pop()` values out of `p` rather than using `p.get()` then passing `**p` — leftover keys cause unexpected keyword argument errors.

---

### 5. Create a feature branch

Always branch off `main`. Use a short, descriptive name:

```bash
git checkout main
git pull upstream main           # sync with the latest upstream first
git checkout -b feat/rouge-metric
```

Branch naming conventions:
- `feat/` — new feature
- `fix/` — bug fix
- `docs/` — documentation only
- `refactor/` — code cleanup with no behaviour change

---

### 6. Make your changes

#### Adding a new evaluation metric

1. Open `create_metric_fn()` (~line 86). Add a new `elif` branch:

```python
elif metric_type == "ROUGE-L":
    from rouge_score import rouge_scorer as _rs
    _scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
    def metric(example, prediction, trace=None):
        gold = _get(example, target_col)
        pred = _get(prediction, target_col)
        return _scorer.score(gold, pred)["rougeL"].fmeasure
```

2. Add the new name to the `st.selectbox` list in Tab 2 (~line 543):

```python
"ROUGE-L",
```

3. Add the new package to `requirements.txt`:

```
rouge-score>=0.1.2
```

#### Adding a new optimizer

1. Add an entry to the `OPTIMIZERS` dict (~line 567). Follow the exact same structure as existing entries — `desc`, `params` (each with `type`, `default`, `min`/`max` or `options`, `help`), and `needs_valset`:

```python
"MyOptimizer": {
    "desc": "One-sentence description shown in the UI info box.",
    "params": {
        "my_param": dict(type="int", default=10, min=1, max=50,
                         help="What this param does."),
    },
    "needs_valset": True,   # set False if compile() does not accept valset
},
```

2. Add a corresponding `elif` branch to `dispatch_optimizer()` (~line 139):

```python
elif opt_name == "MyOptimizer":
    my_p = int(p.pop("my_param", 10))
    opt  = dspy.MyOptimizer(metric=metric, my_param=my_p)
    return opt.compile(module, trainset=trainset, valset=devset)
```

3. Add a cost estimate branch to `_estimate_llm_calls()` (~line 335):

```python
elif opt_name == "MyOptimizer":
    total = n_train * int(params.get("my_param", 10)) + n_eval
```

4. Run the app, switch to Compare mode, tick your new optimizer, and verify it appears in the cost estimate table and runs without errors.

#### Adding a new LLM provider to the sidebar

In the sidebar block (~line 282), add a new `elif` branch:

```python
elif provider == "Groq":
    model       = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"])
    lm_model_id = f"groq/{model}"
```

Also add the provider name to the `st.selectbox` options list above it.

---

### 7. Test your changes manually

There are no automated tests yet (contributing one would itself be a great PR). Until then, verify manually:

1. **Load the built-in example** (10-row training set, 5-row test set) and run your changed feature through the full flow: Dataset → Variables → Optimization → Results.
2. **Upload a real CSV** (use `squad_sample_200.csv` included in the repo) and repeat the flow.
3. **Check edge cases:**
   - Column names with spaces (e.g., rename a column to `"first name"` in the paste editor and confirm sanitization works)
   - Very small datasets (< 10 rows) — the guardrail `min_value` clamping must not crash
   - Missing values in the CSV — the `_to_examples()` function skips NaN cells; ensure your change does not break this
4. **Check the Results tab** — confirm the comparison table, bar chart, per-run inspector, JSON download, and live inference all still work after your change.

---

### 8. Commit your changes

Write a concise commit message that explains *why*, not just *what*:

```bash
git add app.py requirements.txt   # stage only the files you changed
git commit -m "feat: add ROUGE-L metric for summarization tasks

Adds rouge-score based ROUGE-L F-measure to the metric selector.
Useful for tasks where the target is a sentence rather than a short phrase,
since Exact Match and F1 Token Overlap are too strict for those cases."
```

Commit message guidelines:
- First line: `type: short description` (50 chars max), e.g. `feat:`, `fix:`, `docs:`, `refactor:`
- Blank line, then a paragraph explaining the motivation if needed
- Keep each commit focused on one logical change — don't bundle a bug fix and a new feature in one commit

---

### 9. Push and open a pull request

```bash
git push origin feat/rouge-metric
```

Then open a pull request on GitHub from your fork's branch to `main` on the upstream repo.

**PR description template** — fill in all sections:

```
## What this PR does
One sentence summary of the change.

## Why
Explain the motivation. Link to an issue if one exists.

## How to test it
Step-by-step instructions for the reviewer to reproduce your change:
1. Load the built-in example dataset
2. Go to Tab 2, select "ROUGE-L" from the Metric dropdown
3. Run BootstrapFewShot
4. Verify the score in the Results tab is between 0 and 1

## Checklist
- [ ] Tested with the built-in example (10-row dataset)
- [ ] Tested with squad_sample_200.csv
- [ ] requirements.txt updated (if new package added)
- [ ] No API keys or personal data committed
```

---

### 10. Keeping your fork in sync

While your PR is under review, upstream `main` may receive other changes. Rebase your branch to keep a clean history:

```bash
git fetch upstream
git rebase upstream/main feat/rouge-metric
git push origin feat/rouge-metric --force-with-lease
```

Use `--force-with-lease` rather than `--force` — it refuses to overwrite if someone else has pushed to your branch since your last fetch.

---

### Contribution ideas

Pick any of the items below. Difficulty is approximate.

#### Good first issues — estimated 1–2 hours

| Idea | Where to look in app.py |
|------|------------------------|
| Add ROUGE / BLEU / BERTScore metrics | `create_metric_fn()` ~line 86, metric selectbox ~line 543 |
| Add Groq, Gemini, Mistral, Cohere to the sidebar | Sidebar provider selectbox ~line 285 |
| Add more built-in example datasets (medical QA, legal text, code review) | `EXAMPLE_TRAIN` / `EXAMPLE_TEST` dicts ~line 445 |
| Fix the "Always True" metric warning — it should tell the user scores will be meaningless | Results tab, `all_same` warning block ~line 1069 |
| Show a "copy to clipboard" button next to the JSON viewer | Results tab, JSON section ~line 1177 |

#### Medium complexity — estimated half a day

| Idea | Notes |
|------|-------|
| Session save / load | Serialize `st.session_state["optimization_runs"]` to JSON; add a file download and `st.file_uploader` to restore it |
| Chain-of-Thought toggle | Add a checkbox in Tab 2; swap `dspy.Predict` → `dspy.ChainOfThought` in `make_dspy_module()`; surface `rationale` in the Results inspector |
| Token usage tracker | Hook into LiteLLM's `success_callback` to count tokens per run; display tokens + estimated cost in the comparison table |
| Multi-output signatures | Allow multiple target columns in the Variables tab; build signatures like `question -> answer, rationale`; update `create_metric_fn()` to accept a list of targets |
| Custom metric code editor | Embed `st.text_area` with a Python snippet; use `exec()` in a sandbox to define the metric function; validate its signature before use |
| Streaming inference | Replace `st.success()` in the Live Inference section with DSPy's streaming API + `st.write_stream()` |
| Per-run notes | Add a `st.text_input` in the Results inspector; store notes in the run dict; include them in the downloaded JSON |

#### Larger features — estimated several days

| Idea | Notes |
|------|-------|
| Async optimizer runs | Offload each optimizer to `concurrent.futures.ThreadPoolExecutor`; use `st.status()` for live per-optimizer progress; allow multiple optimizers to run in parallel |
| Hyperparameter sweep | For MIPROv2 / Optuna optimizers, add a sweep mode with range inputs; plot score vs. LLM-call cost as a pareto scatter chart |
| RAG pipeline mode | Add a second module type that chains `dspy.Retrieve` + `dspy.ChainOfThought`; let the user configure a retriever (BM25 via `bm25s`, ChromaDB); optimize the full pipeline |
| Export to OpenAI / LangChain format | Add download buttons that convert the optimized prompt + demos to an OpenAI `messages` array or a LangChain `ChatPromptTemplate` |
| Hugging Face Hub push | Add a "Push to Hub" button that uploads the prompt JSON as a HF dataset card using `huggingface_hub` |
| Multi-step pipeline builder | UI to chain multiple Predict/ChainOfThought nodes; serialize the graph; build the composed `dspy.Module` dynamically |

#### Known bugs / limitations to fix

| Bug | Location |
|-----|----------|
| GEPA `reflection_lm` should ideally be a separate, stronger model — add a sidebar picker for it | `dispatch_optimizer()` ~line 202 |
| Cost estimator is based on heuristics — instrument LiteLLM callbacks for actual token counts | `_estimate_llm_calls()` ~line 335 |
| Very large CSVs (>50 MB) hit Streamlit's upload limit — add a "load from local file path" text input as an alternative | `_df_uploader()` ~line 382 |
| `BootstrapFewShot` passes `**p` directly to the constructor — any unexpected key from the params dict will crash it | `dispatch_optimizer()` ~line 144 |

---

### Code style guidelines

- **No comments explaining what the code does** — good names do that. Only comment when explaining *why* something non-obvious is done (a DSPy quirk, a workaround, a hidden constraint).
- **Don't add abstractions for hypothetical future use** — if you only need one optimizer to behave differently, add a targeted `if` rather than a new class hierarchy.
- **Keep the single-file structure** — `app.py` is intentionally self-contained. Don't split into modules unless a feature genuinely requires it (e.g., a separate retriever module for RAG).
- **All new packages go in `requirements.txt`** with a `>=` lower bound, not an exact pin.
- **Never commit API keys, `.env` files, or personal CSV data.** The `.gitignore` blocks `*.csv` and `.env` already, but double-check before pushing.

---

## License

MIT
