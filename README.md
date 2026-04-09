# Occursus Benchmark v2.0 — Multi-Model LLM Benchmark

A benchmarking platform that tests whether multi-LLM synthesis pipelines produce better results than single-model baselines. Supports 4 providers, 22 orchestration strategies, dual blind judging, and subscription CLI mode ($0 cost).

https://dev.to/lam8da/i-built-a-tool-to-test-whether-multiple-llms-working-together-can-beat-a-single-model-4g0l

## Thesis

> Can combining multiple LLMs through structured pipelines (debate, merge, verification, decomposition) consistently outperform a single frontier model answering directly?

## Features

- **4 LLM Providers**: Ollama (local), OpenAI (GPT-4o), Anthropic (Claude), Google Gemini
- **22 Pipeline Architectures**: From single-call baseline to 13-call graph-mesh collaboration
- **Dual Blind Judging**: Claude + GPT judge independently on a 0-100 scale, scores averaged
- **Auto Model Assignment**: Toggle models on/off; the tool assigns them to pipeline roles with multi-model diversity
- **Real-Time Streaming**: Server-Sent Events show results as they complete with buffered replay
- **4 Task Suites**: Smoke (quick), Core (standard), Stress (hard), Thesis (ceiling-breaking)
- **6 Enhancement Toggles**: Chain-of-Thought, Token Budget, Adaptive Temperature, Repeat Runs, Cost Tracking, Subscription CLI Mode
- **Subscription CLI Mode**: Route calls through `claude -p`, `codex exec`, `gemini -p` at $0 extra cost
- **Settings UI**: Configure API keys, base URLs, and test provider connections from the browser
- **Export & History**: CSV/JSON export, persistent run history, rerun failed cells
- **Shutdown Button**: Clean server shutdown from the UI

## Pipeline Tiers

| Tier | Pipelines | Strategy |
|------|-----------|----------|
| 1 | Single, Best of 3, Sample & Vote | Baseline and simple selection |
| 2 | Merge Full, Critique Then Merge, Ranked Merge | Multi-persona synthesis (multi-model) |
| 3 | Debate, Dissent, Red Team/Blue Team, Expert Routing, Constraint Checker | Adversarial and specialist |
| 4 | Chain of Verification, Iterative Refinement, Mixture of Agents, Self-MoA, Adaptive Debate, Reflexion | Deep reasoning and research-backed |
| 5 | Persona Council, Adversarial Decomposition, Reverse Engineer, Tournament, Graph-Mesh Collab | Experimental heavy orchestration |

Research-backed pipelines:
- **Self-MoA** (Princeton 2025): Same-model temperature diversity vs multi-model diversity
- **Adaptive Debate / A-HMAD** (2025): Specialist debaters (+13.2% published results)
- **Reflexion** (2023+): Verbal self-reflection memory (>18% accuracy gains)
- **Graph-Mesh** (MultiAgentBench ACL 2025): All-to-all communication topology

## Two Provider Modes

### API Mode (Default)
Standard REST API calls using your API keys. Full control over temperature, token limits, and concurrency.

### Subscription CLI Mode
Routes calls through your existing paid subscriptions at $0 extra cost:
- Claude via `claude -p` (Anthropic Pro/Max subscription)
- ChatGPT via `codex exec` (OpenAI subscription)
- Gemini via `gemini -p` (Google subscription)

## Enhancement Toggles

| Toggle | Default | What it does |
|--------|---------|-------------|
| Chain-of-Thought | OFF | Forces step-by-step reasoning in generation steps |
| Token Budget | OFF | Reserves 60% of token budget for final synthesis |
| Adaptive Temperature | OFF | Auto-sets temperature based on task type |
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
role_assigner.py      Auto-assigns enabled models to pipeline roles
judge.py              Dual blind judge (Claude + GPT), 0-100 scale
cost_tracker.py       Per-token cost estimation for all providers
task_classifier.py    Regex-based task categorization for adaptive temperature
config.py             YAML + JSON settings, API key management
models.py             Immutable dataclasses (frozen)
db.py                 SQLite persistence with WAL mode
launcher.py           Windows exe launcher with auto-browser
pipelines/
  base.py             Abstract pipeline interface with CoT, token budget, diversity helpers
  single.py           Tier 1: Single-call baseline
  best_of_n.py        Tier 1: Selection strategies
  merge.py            Tier 2: Multi-persona synthesis (multi-model diversity)
  debate.py           Tier 3: Adversarial debate
  routing.py          Tier 3: Expert routing + constraint checker
  deep.py             Tier 4: Verification, refinement, MoA (multi-model)
  experimental.py     Tier 5: Council (multi-model), decomposition, tournament
  research.py         Tier 4-5: Self-MoA, Adaptive Debate, Reflexion, Graph-Mesh
static/
  index.html          Single-page UI with provider mode, toggles, model selection
  app.js              Frontend logic (SSE streaming, charts, modals)
  style.css           Dark theme
tasks/                JSON task definition files (smoke, core, stress, thesis)
tests/                Test suite
```

## License

MIT
