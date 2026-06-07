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

Pull requests are welcome. Below are ideas ranging from small to ambitious.

### Good first issues

- **Add a new metric** — edit `create_metric_fn()` to add ROUGE, BLEU, BERTScore, or a regex-based metric. Also add it to the metric selectbox in Tab 2.
- **Add a new built-in example dataset** — extend the `EXAMPLE_TOPICS` dict in `_df_uploader()` with a new topic and a `pd.DataFrame` of rows.
- **Persist results to disk** — add a "Save session" / "Load session" button that serializes `st.session_state["optimization_runs"]` to a JSON file so runs survive a page refresh.
- **Show token usage** — after each optimizer run, display the total tokens consumed using `dspy.settings.usage_tracker` or the LiteLLM callback.
- **Dark-mode chart styling** — the Altair/Streamlit bar chart currently uses default colors; match it to Streamlit's dark theme when active.

### Medium complexity

- **Support multi-output signatures** — the current signature builder forces a single target column. Extend `make_dspy_module()` and the Variables tab to support multiple output fields (e.g., `answer, rationale`).
- **Add a Chain-of-Thought toggle** — replace `dspy.Predict` with `dspy.ChainOfThought` when the user enables it, and surface the rationale field in the results inspector.
- **Custom metric via code editor** — embed a small `st.code_editor` (or `st_ace`) widget that lets the user write a Python metric function directly in the UI, with live syntax checking.
- **Per-predictor instruction editing** — let the user manually edit the proposed instructions in the Results tab and re-evaluate without re-running the full optimizer.
- **Streaming LLM output** — use DSPy's streaming API so the Live Inference section shows tokens appearing in real time rather than waiting for the full response.

### Larger features

- **Multi-step pipeline support** — allow the user to chain multiple Predict/ChainOfThought modules (e.g., retrieve → reason → answer) and optimize the full pipeline. This would require a visual graph builder in the UI.
- **Retrieval-augmented generation (RAG)** — integrate `dspy.Retrieve` with a configurable retriever (BM25 via `bm25s`, ChromaDB, Pinecone) so the app can optimize RAG pipelines end-to-end.
- **Hyperparameter sweep UI** — for MIPROv2 and Optuna-based optimizers, add a range-sweep mode that tries multiple parameter combinations and plots the pareto frontier of score vs. cost.
- **Remote / async execution** — long optimizer runs block the Streamlit session. Offload optimizer runs to a background thread or Celery worker and poll for completion with a progress bar.
- **Export to LangChain / OpenAI format** — in addition to the DSPy JSON, add export buttons that convert the optimized prompt into a LangChain `ChatPromptTemplate` or an OpenAI `messages` array.
- **Hugging Face Hub integration** — push the exported prompt JSON to a HF dataset repo so prompts are versioned and shareable.

### Known limitations to fix

- GEPA's `reflection_lm` defaults to the same LM used for inference. Adding a separate model picker for reflection (ideally a stronger, higher-temperature model) would improve its output quality.
- The cost estimator uses rough heuristics. A more accurate version would instrument LiteLLM callbacks and report actual token counts.
- Very large CSVs (>50 MB) can exhaust Streamlit's upload limit. Adding a "load from local path" text input would work around this.

---

## License

MIT
