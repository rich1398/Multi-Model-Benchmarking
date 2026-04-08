# Occursus-Claude — Multi-Model LLM Benchmark

A benchmarking platform that tests whether multi-LLM synthesis pipelines produce better results than single-model baselines. Supports 4 providers (Ollama, OpenAI, Anthropic, Gemini) with 18 orchestration strategies and dual blind judging.

## Thesis

> Can combining multiple LLMs through structured pipelines (debate, merge, verification, decomposition) consistently outperform a single frontier model answering directly?

## Features

- **4 LLM Providers**: Ollama (local), OpenAI (GPT-4o), Anthropic (Claude), Google Gemini
- **18 Pipeline Architectures**: From single-call baseline to 12-call tournament elimination
- **Dual Blind Judging**: Claude + GPT judge independently on a 0-100 scale, scores averaged
- **Auto Model Assignment**: Toggle models on/off; the tool assigns them to pipeline roles automatically
- **Real-Time Streaming**: Server-Sent Events show results as they complete
- **4 Task Suites**: Smoke (quick), Core (standard), Stress (hard), Thesis (ceiling-breaking)
- **Settings UI**: Configure API keys, base URLs, and test provider connections from the browser
- **Export & History**: CSV/JSON export, persistent run history, rerun failed cells

## Pipeline Tiers

| Tier | Pipelines | Strategy |
|------|-----------|----------|
| 1 | Single, Best of 3, Sample & Vote | Baseline and simple selection |
| 2 | Merge Full, Critique Then Merge, Ranked Merge | Multi-persona synthesis |
| 3 | Debate, Dissent, Red Team/Blue Team, Expert Routing, Constraint Checker | Adversarial and specialist |
| 4 | Chain of Verification, Iterative Refinement, Mixture of Agents | Deep reasoning loops |
| 5 | Persona Council, Adversarial Decomposition, Reverse Engineer, Tournament | Experimental heavy |

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) (optional, for local models)
- API keys for cloud providers (optional)

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

Open `http://localhost:8000` in your browser.

### Configure API Keys

1. Open the **Provider Settings** panel in the UI
2. Enter API keys for OpenAI, Anthropic, and/or Gemini
3. Click **Test** to verify each connection
4. Click **Save Settings**

Or use environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Task Suites

| Suite | Tasks | Difficulty | Purpose |
|-------|-------|------------|---------|
| `smoke_tasks.json` | 5 | Easy | Quick validation |
| `core_tasks.json` | 12 | Easy-Medium | Standard benchmark |
| `stress_tasks.json` | 8 | Hard | Complex reasoning |
| `thesis_tasks.json` | 8 | Very Hard | Ceiling-breaking tasks designed to differentiate pipelines |

The **thesis tasks** include cross-domain synthesis, multi-file code refactoring, constraint satisfaction, needle-in-haystack analysis, and multi-requirement system design — problems where single models demonstrably struggle.

## Architecture

```
app.py              FastAPI server, API endpoints, benchmark orchestration
llm_client.py       Unified async client for all 4 providers, auto-routing, retry logic
role_assigner.py    Auto-assigns enabled models to pipeline roles
judge.py            Dual blind judge (Claude + GPT), 0-100 scale
config.py           YAML + JSON settings, API key management
models.py           Immutable dataclasses (frozen)
db.py               SQLite persistence with WAL mode
pipelines/
  base.py           Abstract pipeline interface
  single.py         Tier 1: Single-call baseline
  best_of_n.py      Tier 1: Selection strategies
  merge.py          Tier 2: Multi-persona synthesis
  debate.py         Tier 3: Adversarial debate
  routing.py        Tier 3: Expert routing + constraint checker
  deep.py           Tier 4: Verification, refinement, MoA
  experimental.py   Tier 5: Council, decomposition, tournament
static/
  index.html        Single-page UI
  app.js            Frontend logic (SSE streaming, charts, modals)
  style.css         Dark theme
tasks/              JSON task definition files
```

## How It Works

1. **Toggle models** on/off in the UI (6 preset models across 4 providers)
2. **Select pipelines** and tasks to benchmark
3. **Click Run** — the tool auto-assigns models to roles:
   - Claude/GPT as primary generator and synthesizer
   - Secondary cloud model as critic and alternative generator
   - Gemini for diversity in multi-model pipelines
   - Ollama for speed when many models are enabled
4. **Dual judge** — both Claude and GPT score each response blind on 0-100, averaged
5. **View results** — real-time score matrix, charts, drill-down, thesis banner

## License

MIT
