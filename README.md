# Occursus Benchmark v2.0 — Multi-Model LLM Benchmark

A benchmarking platform that tests whether multi-LLM synthesis pipelines produce better results than single-model baselines. Supports 4 providers, 29 orchestration strategies, dual blind judging, flagship models, and subscription CLI mode ($0 cost).

https://dev.to/lam8da/i-built-a-tool-to-test-whether-multiple-llms-working-together-can-beat-a-single-model-4g0l

## Thesis

> Can combining multiple LLMs through structured pipelines (debate, merge, verification, decomposition) consistently outperform a single frontier model answering directly?

## Features

- **4 LLM Providers**: Ollama (local), OpenAI (GPT-5.4), Anthropic (Claude Opus 4.6), Google Gemini 2.5 Pro
- **29 Pipeline Architectures**: From single-call baseline to 17-call full combination pipelines
- **Dual Blind Judging**: Claude Opus 4.6 + GPT-5.4 judge independently on a 0-100 scale, scores averaged
- **Flagship Models**: Uses the strongest available model from each provider by default
- **Multi-Model Diversity**: Pipelines cycle through Claude/GPT/Gemini for true cross-model collaboration
- **Real-Time Streaming**: Server-Sent Events show results as they complete with buffered replay
- **4 Task Suites**: Smoke (quick), Core (standard), Stress (hard), Thesis (ceiling-breaking)
- **6 Enhancement Toggles**: Chain-of-Thought, Token Budget, Adaptive Temperature, Repeat Runs, Cost Tracking, Subscription CLI Mode
- **Two Provider Modes**: API (full control) or Subscription CLI ($0 extra cost)
- **Settings UI**: Configure API keys, base URLs, and test provider connections from the browser
- **Export & History**: CSV/JSON export, persistent run history, rerun failed cells
- **Shutdown Button**: Clean server shutdown from the UI
- **Windows Executable**: Standalone `Occursus-Benchmark.exe` with auto-browser launch

## Pipeline Tiers (29 pipelines)

| Tier | Pipelines | Strategy |
|------|-----------|----------|
| **1 — Baseline** | Single, Best of 3, Sample & Vote | Direct call and simple selection |
| **2 — Synthesis** | Merge Full, Critique Then Merge, Ranked Merge | Multi-persona generation + synthesis (multi-model) |
| **3 — Adversarial** | Debate, Dissent, Red Team/Blue Team, Expert Routing, Constraint Checker | Models challenge each other's work |
| **4 — Deep** | Chain of Verification, Iterative Refinement, Mixture of Agents, Self-MoA, Adaptive Debate, Reflexion, Adaptive Cascade, Managed Team | Research-backed reasoning loops and corporate structure |
| **5 — Experimental** | Persona Council, Adversarial Decomposition, Reverse Engineer, Tournament, Graph-Mesh, Mesh+Verify, Mesh+Ranked, GSV, Mesh+Ranked+Verify, Corp Hierarchy | Heavy orchestration and combination pipelines |

### Research-backed pipelines
- **Self-MoA** (Princeton 2025): Same-model temperature diversity vs multi-model diversity
- **Adaptive Debate / A-HMAD** (2025): Specialist debaters (+13.2% published results)
- **Reflexion** (2023+): Verbal self-reflection memory (>18% accuracy gains)
- **Graph-Mesh** (MultiAgentBench ACL 2025): All-to-all communication topology

### Combination pipelines (top-3 architecture fusion)
- **Mesh + Verify**: Graph-mesh collaboration → structured verification → fix
- **Mesh + Ranked**: Graph-mesh → pairwise tournament selection
- **Generate-Select-Verify (GSV)**: 5 diverse candidates → tournament → verify → fix
- **Mesh+Ranked+Verify**: Full combination — all three mechanisms
- **Adaptive Cascade**: Verify-first — cheap path (2 calls) if passes, graph-mesh if fails

### Corporate hierarchy pipelines
- **Managed Team**: Lead decomposes → specialists execute (multi-model) → critic reviews → verifier checks
- **Corp Hierarchy**: 4-tier adaptive — T0 local, T1 flagship, T2 managed team, T3 multi-clique deliberation

## Two Provider Modes

### API Mode (Default)
Standard REST API calls using your API keys. Full control over temperature, token limits, and concurrency. Uses flagship models: Claude Opus 4.6, GPT-5.4, Gemini 2.5 Pro.

### Subscription CLI Mode
Routes calls through your existing paid subscriptions at $0 extra cost:
- Claude via `claude -p --model opus` (Anthropic Pro/Max subscription)
- ChatGPT via `codex exec` (OpenAI subscription)
- Gemini via `gemini -p --model gemini-2.5-pro` (Google subscription)

## Enhancement Toggles

| Toggle | Default | What it does |
|--------|---------|-------------|
| Chain-of-Thought | OFF | Forces step-by-step reasoning in generation steps |
| Token Budget | OFF | Reserves 60% of token budget for final synthesis |
| Adaptive Temperature | OFF | Auto-sets temperature based on task type (factual/code/analytical/creative) |
| Repeat Runs | 1x | Run each cell 1/3/5 times for statistical significance |
| Cost Tracking | ON | Estimates and displays $/pipeline using published pricing |

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) (optional, for local models)
- API keys for cloud providers (optional — or use Subscription CLI mode)

### Install

```bash
git clone https://github.com/rich1398/Multi-Model-Benchmarking.git
cd Multi-Model-Benchmarking
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Or double-click `Occursus-Benchmark.exe` (Windows).

Open `http://localhost:8000` in your browser.

### Configure

1. Open the **Provider Settings** panel
2. Enter API keys for OpenAI, Anthropic, and/or Gemini (or switch to Subscription CLI mode)
3. Click **Test** to verify each connection
4. Toggle models on/off, select pipelines and tasks, click **Run Benchmark**

## Architecture

```
app.py                FastAPI server, API endpoints, benchmark orchestration
llm_client.py         Unified async client (4 API providers + 3 subscription CLIs)
role_assigner.py      Auto-assigns enabled models to pipeline roles with diversity
judge.py              Dual blind judge (Claude Opus 4.6 + GPT-5.4), 0-100 scale
cost_tracker.py       Per-token cost estimation for all providers
task_classifier.py    Regex-based task categorization for adaptive temperature
config.py             YAML + JSON settings, API key management
models.py             Immutable dataclasses (frozen)
db.py                 SQLite persistence with WAL mode
launcher.py           Windows exe launcher with auto-browser
pipelines/
  base.py             Abstract pipeline with CoT, token budget, diversity helpers
  single.py           Tier 1: Single-call baseline
  best_of_n.py        Tier 1: Selection strategies
  merge.py            Tier 2: Multi-persona synthesis (multi-model diversity)
  debate.py           Tier 3: Adversarial debate
  routing.py          Tier 3: Expert routing + constraint checker
  deep.py             Tier 4: Verification, refinement, MoA (multi-model)
  research.py         Tier 4-5: Self-MoA, Adaptive Debate, Reflexion, Graph-Mesh
  combinations.py     Tier 5: Mesh+Verify, Mesh+Ranked, GSV, full combo, cascade
  hierarchy.py        Tier 4-5: Managed Team, Corp Hierarchy (4-tier adaptive)
  experimental.py     Tier 5: Council, decomposition, tournament
static/
  index.html          Single-page UI with provider mode, toggles, model selection
  app.js              Frontend logic (SSE streaming, charts, modals)
  style.css           Dark theme
tasks/                JSON task suites (smoke, core, stress, thesis)
tests/                Test suite
```

## Early Results

On hard thesis tasks with flagship models and strict dual judging (Opus 4.6 + GPT-5.4):

- **Graph-Mesh Collaboration** led earlier benchmarks at 93.3 avg (+10.8 vs baseline)
- **Chain of Verification** strong at 91.3 (+8.8 vs baseline)
- **Ranked Merge** consistent at 91.2 (+8.7 vs baseline)
- **Selection-based pipelines** (Sample & Vote, Ranked Merge) consistently outperform synthesis-based approaches
- **Debate-style pipelines** underperform on constrained tasks
- Multi-model pipelines show **genuine, measurable improvement** over single-model on hard tasks

Combination pipelines and corporate hierarchy variants are currently being benchmarked.

## License

MIT
