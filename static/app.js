/* Occursus Benchmark Frontend */

const BENCHMARK_MODELS = [
  { provider: 'ollama',    model: 'llama3.2',                label: 'Llama 3.2',         badge: 'local' },
  { provider: 'ollama',    model: 'mistral',                 label: 'Mistral 7B',        badge: 'local' },
  { provider: 'ollama',    model: 'qwen3:8b',                label: 'Qwen3 8B',          badge: 'local' },
  { provider: 'openai',    model: 'gpt-5.4',                 label: 'GPT-5.4',           badge: 'openai' },
  { provider: 'anthropic', model: 'claude-opus-4-6',          label: 'Claude Opus 4.6',   badge: 'anthropic' },
  { provider: 'gemini',    model: 'gemini-2.5-pro',          label: 'Gemini 2.5 Pro',    badge: 'gemini' },
];

const state = {
  pipelines: [],
  tasks: [],
  suites: [],
  models: [],
  providerKeys: {},
  defaultModel: 'llama3.2',
  currentRunId: null,
  currentRunDetails: null,
  activeRunId: null,
  results: [],
  chart: null,
  eventSource: null,
  savedGenerationModels: [],
};

document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await loadSettings();
  await Promise.all([
    loadModels(),
    loadPipelines(),
    loadTaskSuites(),
    loadHistory(),
  ]);
});

async function loadModels() {
  try {
    const resp = await fetch('/api/models');
    const data = await resp.json();
    state.defaultModel = data.default_model || 'llama3.2';
    state.providerKeys = data.provider_keys || {};
  } catch (e) {
    console.error('Failed to load models:', e);
    state.providerKeys = {};
  }
  renderModelToggles();
  updateHealthBanner();
}

function renderModelToggles() {
  const container = document.getElementById('model-toggles');
  if (!container) return;
  const savedSet = new Set(
    (state.savedGenerationModels || []).map(m => `${m.provider}:${m.model}`)
  );
  const defaultAll = savedSet.size === 0;

  let html = '';
  BENCHMARK_MODELS.forEach(bm => {
    const key = `${bm.provider}:${bm.model}`;
    const isCloud = bm.provider !== 'ollama';
    const hasKey = isCloud ? state.providerKeys[bm.provider] : true;
    const checked = (defaultAll && !isCloud) || savedSet.has(key) ? ' checked' : '';
    const disabled = isCloud && !hasKey ? ' disabled' : '';
    const hint = disabled ? ' <span class="dim">(no API key)</span>' : '';
    const badgeClass = isCloud ? `badge-${bm.badge}` : 'badge-local';

    html += `<label class="model-toggle-label${disabled ? ' disabled' : ''}">
      <input type="checkbox" class="model-toggle-cb"
             data-provider="${bm.provider}" data-model="${escapeHtml(bm.model)}"
             ${checked}${disabled}>
      <span class="badge ${badgeClass}">${escapeHtml(bm.badge)}</span>
      <span>${escapeHtml(bm.label)}${hint}</span>
    </label>`;
  });
  container.innerHTML = html;

  container.addEventListener('change', updateHealthBanner);
}

/* --- Settings Management --- */

async function loadSettings() {
  try {
    const resp = await fetch('/api/settings');
    const data = await resp.json();
    document.getElementById('setting-ollama-url').value = data.ollama_base_url || 'http://localhost:11434';
    document.getElementById('setting-openai-url').value = data.openai_base_url || 'https://api.openai.com';
    document.getElementById('setting-anthropic-url').value = data.anthropic_base_url || 'https://api.anthropic.com';
    document.getElementById('setting-gemini-url').value = data.gemini_base_url || 'https://generativelanguage.googleapis.com';
    if (data.has_openai_key) document.getElementById('setting-openai-key').value = data.openai_api_key;
    if (data.has_anthropic_key) document.getElementById('setting-anthropic-key').value = data.anthropic_api_key;
    if (data.has_gemini_key) document.getElementById('setting-gemini-key').value = data.gemini_api_key;
    state.savedGenerationModels = data.selected_generation_models || [];
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
}

async function saveSettings() {
  const statusEl = document.getElementById('settings-save-status');
  statusEl.textContent = 'Saving...';
  try {
    const enabledModels = [...document.querySelectorAll('.model-toggle-cb:checked')].map(cb => ({
      provider: cb.dataset.provider,
      model: cb.dataset.model,
      label: `${cb.dataset.provider}:${cb.dataset.model}`,
    }));
    const body = {
      ollama_base_url: document.getElementById('setting-ollama-url').value,
      openai_base_url: document.getElementById('setting-openai-url').value,
      anthropic_base_url: document.getElementById('setting-anthropic-url').value,
      gemini_base_url: document.getElementById('setting-gemini-url').value,
      openai_api_key: document.getElementById('setting-openai-key').value,
      anthropic_api_key: document.getElementById('setting-anthropic-key').value,
      gemini_api_key: document.getElementById('setting-gemini-key').value,
      selected_generation_models: enabledModels,
    };
    const resp = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    statusEl.textContent = data.ok ? 'Saved!' : (data.message || 'Error');
    if (data.ok) {
      await loadModels();
      setTimeout(() => { statusEl.textContent = ''; }, 3000);
    }
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  }
}

async function testProvider(provider) {
  const statusEl = document.getElementById(`status-${provider}`);
  statusEl.textContent = 'Testing...';
  statusEl.className = 'provider-status';
  try {
    const resp = await fetch('/api/settings/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        provider,
        api_key: document.getElementById(`setting-${provider}-key`)?.value || '',
        base_url: document.getElementById(`setting-${provider}-url`)?.value || '',
      }),
    });
    const data = await resp.json();
    statusEl.textContent = data.ok
      ? `OK: ${(data.models || []).length} models`
      : `Failed: ${data.message || 'Unknown error'}`;
    statusEl.className = 'provider-status ' + (data.ok ? 'ok' : 'error');
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    statusEl.className = 'provider-status error';
  }
}

function getEnabledModels() {
  return [...document.querySelectorAll('.model-toggle-cb:checked')].map(cb => ({
    provider: cb.dataset.provider,
    model: cb.dataset.model,
  }));
}

async function loadPipelines() {
  try {
    const resp = await fetch('/api/pipelines');
    const data = await resp.json();
    state.pipelines = data.pipelines || [];
    renderPipelineCheckboxes();
  } catch (e) {
    console.error('Failed to load pipelines:', e);
  }
}

async function loadTaskSuites() {
  const select = document.getElementById('task-suite-select');
  try {
    const resp = await fetch('/api/task-suites');
    const data = await resp.json();
    state.suites = data.suites || [];
    if (!state.suites.length) {
      state.suites = [{ id: 'tasks.json', name: 'tasks' }];
    }
    select.innerHTML = state.suites.map((suite) =>
      `<option value="${suite.id}">${escapeHtml(suite.name)}</option>`
    ).join('');
  } catch (e) {
    console.error('Failed to load task suites:', e);
    select.innerHTML = '<option value="tasks.json">tasks</option>';
  }
  await loadTasks();
}

async function loadTasks() {
  try {
    const suite = document.getElementById('task-suite-select').value;
    const resp = await fetch(`/api/tasks?suite=${encodeURIComponent(suite)}`);
    const data = await resp.json();
    state.tasks = data.tasks || [];
    renderTaskCheckboxes();
  } catch (e) {
    console.error('Failed to load tasks:', e);
  }
}

async function loadHistory() {
  try {
    const resp = await fetch('/api/results');
    const data = await resp.json();
    renderHistory(data.runs || []);
  } catch (e) {
    console.error('Failed to load history:', e);
  }
}

function renderPipelineCheckboxes() {
  const container = document.getElementById('pipeline-checkboxes');
  const tiers = {};
  state.pipelines.forEach((p) => {
    if (!tiers[p.tier]) tiers[p.tier] = [];
    tiers[p.tier].push(p);
  });

  const tierNames = {
    1: 'Tier 1: Baseline & Simple',
    2: 'Tier 2: Practical Synthesis',
    3: 'Tier 3: Adversarial & Debate',
    4: 'Tier 4: Deep & Slow',
    5: 'Tier 5: Experimental',
  };

  let html = '';
  for (const [tier, pipes] of Object.entries(tiers).sort()) {
    html += `<div class="tier-label">${tierNames[tier] || `Tier ${tier}`}</div>`;
    pipes.forEach((p) => {
      html += `<label title="${escapeHtml(p.description)}">
        <input type="checkbox" name="pipeline" value="${p.id}" checked>
        ${escapeHtml(p.name)} <span style="color:var(--text-dim);font-size:0.75rem">(~${p.estimated_calls} calls)</span>
      </label>`;
    });
  }
  container.innerHTML = html;
}

function renderTaskCheckboxes() {
  const container = document.getElementById('task-checkboxes');
  container.innerHTML = state.tasks.map((task) =>
    `<label title="${escapeHtml(task.rubric)}">
      <input type="checkbox" name="task" value="${task.id}" checked>
      <strong>${escapeHtml(task.id)}</strong>: ${escapeHtml(truncate(task.prompt, 60))}
    </label>`
  ).join('');
}

function renderHistory(runs) {
  const container = document.getElementById('history-list');
  if (!runs.length) {
    container.innerHTML = '<p style="color:var(--text-dim)">No previous runs</p>';
    return;
  }

  container.innerHTML = runs.map((run) => {
    const startedAt = run.started_at
      ? new Date(run.started_at * 1000).toLocaleString()
      : 'Unknown date';
    const completed = run.completed_tasks || run.completed || 0;
    const total = run.total_tasks || run.total || 0;
    const summary = run.summary || {};
    const avgValue = run.average_score_all_tasks != null
      ? run.average_score_all_tasks
      : summary.average_score_all_tasks;
    const successRate = run.success_rate != null
      ? run.success_rate
      : summary.success_rate;
    const avg = avgValue != null
      ? `avg ${Number(avgValue).toFixed(2)}`
      : 'avg -';
    const success = successRate != null
      ? `${Math.round(Number(successRate) * 100)}% success`
      : '';
    return `<div class="history-item" onclick="loadRun('${run.run_id}')">
      <span>Run ${run.run_id} - ${escapeHtml(run.status || 'unknown')}</span>
      <span class="history-meta">${startedAt} - ${completed}/${total} cells - ${avg}${success ? ` - ${success}` : ''}</span>
    </div>`;
  }).join('');
}

function setupEventListeners() {
  document.getElementById('btn-run').addEventListener('click', startRun);
  document.getElementById('btn-stop').addEventListener('click', stopRun);
  document.getElementById('btn-export-csv').addEventListener('click', () => exportResults('csv'));
  document.getElementById('btn-export-json').addEventListener('click', () => exportResults('json'));
  document.getElementById('btn-rerun-failed').addEventListener('click', rerunFailed);
  document.getElementById('modal-close').addEventListener('click', closeModal);
  document.getElementById('modal').addEventListener('click', (e) => {
    if (e.target === document.getElementById('modal')) closeModal();
  });
  document.getElementById('task-suite-select').addEventListener('change', loadTasks);

  document.getElementById('btn-save-settings').addEventListener('click', saveSettings);
  document.getElementById('btn-shutdown').addEventListener('click', shutdownServer);
  document.querySelectorAll('.btn-test').forEach((btn) => {
    btn.addEventListener('click', () => testProvider(btn.dataset.provider));
  });

  // Provider mode toggle — disable incompatible enhancements in subscription mode
  document.querySelectorAll('input[name="provider-mode"]').forEach(radio => {
    radio.addEventListener('change', () => {
      const isSub = document.querySelector('input[name="provider-mode"]:checked')?.value === 'subscription';
      const tokenBudget = document.getElementById('cb-token-budget');
      const adaptiveTemp = document.getElementById('cb-adaptive-temp');
      tokenBudget.disabled = isSub;
      adaptiveTemp.disabled = isSub;
      if (isSub) {
        tokenBudget.checked = false;
        adaptiveTemp.checked = false;
      }
      document.getElementById('toggle-token-budget').classList.toggle('disabled', isSub);
      document.getElementById('toggle-adaptive-temp').classList.toggle('disabled', isSub);
      updateHealthBanner();
    });
  });
}

function updateHealthBanner() {
  const banner = document.getElementById('health-banner');
  const enabled = getEnabledModels();
  const providers = [...new Set(enabled.map(m => m.provider))];
  const hasCloudJudge = enabled.some(m => m.provider === 'anthropic' || m.provider === 'openai');

  if (!enabled.length) {
    banner.className = 'health-banner error';
    banner.textContent = 'Enable at least one model to run benchmarks.';
    return;
  }

  const judgeNote = hasCloudJudge
    ? 'Dual blind judging active (Claude + GPT).'
    : 'No cloud models enabled — local model will judge.';

  banner.className = 'health-banner ok';
  banner.textContent = `Ready: ${enabled.length} models (${providers.join(', ')}). ${judgeNote}`;
}

async function startRun() {
  // Clean up any previous run's SSE connection
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  if (state.activeRunId) {
    fetch(`/api/run/${state.activeRunId}`, { method: 'DELETE' }).catch(() => {});
    state.activeRunId = null;
  }

  const selectedPipelines = [...document.querySelectorAll('input[name="pipeline"]:checked')]
    .map((cb) => cb.value);
  const selectedTasks = [...document.querySelectorAll('input[name="task"]:checked')]
    .map((cb) => cb.value);
  const taskSuite = document.getElementById('task-suite-select').value;
  const customPrompt = document.getElementById('custom-prompt').value.trim();
  const enabledModels = getEnabledModels();
  const effectiveTaskIds = selectedTasks.length
    ? selectedTasks
    : state.tasks.map((task) => task.id);

  if (!enabledModels.length) {
    alert('Enable at least one model');
    return;
  }
  if (!selectedPipelines.length) {
    alert('Select at least one pipeline');
    return;
  }

  const body = {
    pipelines: selectedPipelines,
    enabled_models: enabledModels,
    task_suite: taskSuite,
    subscription_mode: document.querySelector('input[name="provider-mode"]:checked')?.value === 'subscription',
    cot_enabled: document.getElementById('cb-cot')?.checked || false,
    token_budget_enabled: document.getElementById('cb-token-budget')?.checked || false,
    adaptive_temp_enabled: document.getElementById('cb-adaptive-temp')?.checked || false,
    repeat_count: parseInt(document.getElementById('select-repeat')?.value || '1', 10),
    cost_tracking_enabled: document.getElementById('cb-cost-tracking')?.checked || false,
  };

  if (customPrompt) {
    body.custom_prompt = customPrompt;
  } else {
    body.task_ids = effectiveTaskIds.length ? effectiveTaskIds : undefined;
  }

  try {
    document.getElementById('btn-run').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('btn-rerun-failed').disabled = true;

    const resp = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await resp.json();
    if (!resp.ok || data.error) {
      const detailText = (data.details || []).join('\n');
      alert(detailText ? `${data.error}\n\n${detailText}` : (data.error || `HTTP ${resp.status}`));
      resetRunButtons();
      return;
    }

    state.currentRunId = data.run_id;
    state.activeRunId = data.run_id;
    state.currentRunDetails = {
      run_id: data.run_id,
      config: body,
      results: [],
      summary: {},
    };
    state.results = [];

    document.getElementById('results-panel').style.display = 'block';
    document.getElementById('result-run-id').textContent = `(${data.run_id})`;
    document.getElementById('progress-text').textContent = `0/${data.total}`;
    document.getElementById('progress-bar').style.width = '0%';

    initChart(selectedPipelines);
    initTable(selectedPipelines, customPrompt ? ['custom'] : effectiveTaskIds);
    connectSSE(data.run_id, data.total);
  } catch (e) {
    console.error('Failed to start run:', e);
    alert(`Failed to start benchmark: ${e.message}`);
    resetRunButtons();
  }
}

function stopRun() {
  const runToCancel = state.activeRunId || state.currentRunId;
  if (runToCancel) {
    fetch(`/api/run/${runToCancel}`, { method: 'DELETE' });
  }
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.activeRunId = null;
  resetRunButtons();
}

async function shutdownServer() {
  if (!confirm('Shut down the Occursus Benchmark server? All running benchmarks will be cancelled.')) return;
  try {
    stopRun();
    await fetch('/api/shutdown', { method: 'POST' });
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#888;font-size:1.5rem">Server stopped. You can close this tab.</div>';
  } catch (e) {
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#888;font-size:1.5rem">Server stopped. You can close this tab.</div>';
  }
}

function resetRunButtons() {
  document.getElementById('btn-run').disabled = false;
  document.getElementById('btn-stop').disabled = true;
}

function updateActiveRunBanner() {
  let banner = document.getElementById('active-run-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'active-run-banner';
    banner.className = 'active-run-banner';
    const resultsPanel = document.getElementById('results-panel');
    if (resultsPanel) resultsPanel.prepend(banner);
  }

  if (state.activeRunId && state.currentRunId !== state.activeRunId) {
    banner.style.display = 'block';
    banner.innerHTML = `A benchmark is running (${state.activeRunId}). <button class="btn-sm" onclick="returnToActiveRun()">View Live Results</button>`;
  } else {
    banner.style.display = 'none';
  }
}

async function returnToActiveRun() {
  if (!state.activeRunId) return;
  state.currentRunId = state.activeRunId;
  state.results = state.currentRunDetails?.run_id === state.activeRunId
    ? (state.currentRunDetails.results || [])
    : [];

  const pipelineIds = [...new Set(state.results.map(e => e.pipeline_id))];
  const taskIds = [...new Set(state.results.map(e => e.task_id))];

  document.getElementById('results-panel').style.display = 'block';
  document.getElementById('result-run-id').textContent = `(${state.activeRunId})`;
  document.getElementById('progress-text').textContent = 'Live run in progress...';

  if (pipelineIds.length && taskIds.length) {
    initChart(pipelineIds);
    initTable(pipelineIds, taskIds);
    state.results.forEach(entry => updateTableCell(entry));
    updateChart();
    updateStats(currentSummary());
    updateThesisBanner(currentSummary());
  }
  updateActiveRunBanner();
}

function connectSSE(runId, total) {
  if (state.eventSource) state.eventSource.close();

  const es = new EventSource(`/api/stream/${runId}`);
  state.eventSource = es;

  es.addEventListener('progress', (e) => {
    const data = JSON.parse(e.data);
    document.getElementById('progress-text').textContent = data.message || '';
  });

  es.addEventListener('result', (e) => {
    const data = JSON.parse(e.data);
    upsertResult(data);

    const pct = total ? (data.completed / total * 100).toFixed(0) : '0';
    document.getElementById('progress-bar').style.width = `${pct}%`;
    document.getElementById('progress-text').textContent =
      `${data.completed}/${total} - ${data.pipeline_id} on ${data.task_id}: ${data.score}/10`;

    updateChart();
    updateTableCell(data);
    updateThesisBanner(data.summary || currentSummary());
    updateStats(data.summary || currentSummary());
  });

  es.addEventListener('done', (e) => {
    const data = JSON.parse(e.data);
    es.close();
    state.eventSource = null;
    state.activeRunId = null;
    resetRunButtons();
    updateActiveRunBanner();

    document.getElementById('progress-text').textContent =
      data.status === 'completed' ? 'Benchmark complete!' :
      data.status === 'completed_with_failures' ? 'Benchmark finished with failures.' :
      data.status === 'completed_with_judge_failures' ? 'Benchmark finished with judge failures.' :
      data.status === 'cancelled' ? 'Benchmark cancelled.' :
      `Error: ${data.error || 'Unknown'}`;

    document.getElementById('progress-bar').style.width = '100%';
    if (state.currentRunDetails) {
      state.currentRunDetails.summary = data.summary || currentSummary();
      state.currentRunDetails.status = data.status;
    }
    updateThesisBanner(data.summary || currentSummary());
    updateStats(data.summary || currentSummary());
    document.getElementById('btn-rerun-failed').disabled = !(data.summary && data.summary.failure_count > 0);
    loadHistory();
  });

  es.addEventListener('ping', () => {});

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) {
      resetRunButtons();
    }
  };
}

function upsertResult(result) {
  const key = `${result.task_id}::${result.pipeline_id}`;
  const idx = state.results.findIndex((entry) => `${entry.task_id}::${entry.pipeline_id}` === key);
  if (idx >= 0) {
    state.results[idx] = result;
  } else {
    state.results.push(result);
  }

  if (state.currentRunDetails) {
    if (!Array.isArray(state.currentRunDetails.results)) {
      state.currentRunDetails.results = [];
    }
    const detailIdx = state.currentRunDetails.results.findIndex(
      (entry) => `${entry.task_id}::${entry.pipeline_id}` === key
    );
    if (detailIdx >= 0) {
      state.currentRunDetails.results[detailIdx] = result;
    } else {
      state.currentRunDetails.results.push(result);
    }
    state.currentRunDetails.summary = result.summary || currentSummary();
  }
}

function currentSummary() {
  const total = state.results.length;
  const successCount = state.results.filter((entry) => entry.ok).length;
  const failureCount = total - successCount;
  const judgeFailureCount = state.results.filter((entry) => entry.judge_ok === false).length;
  const totalScore = state.results.reduce((sum, entry) => sum + Number(entry.score || 0), 0);
  const successScores = state.results.filter((entry) => entry.ok).map((entry) => Number(entry.score || 0));
  return {
    total_results: total,
    success_count: successCount,
    failure_count: failureCount,
    judge_failure_count: judgeFailureCount,
    success_rate: total ? successCount / total : 0,
    average_score_all_tasks: total ? totalScore / total : 0,
    average_score_success_only: successScores.length
      ? successScores.reduce((sum, score) => sum + score, 0) / successScores.length
      : null,
    valid_for_thesis: total > 0 && failureCount === 0 && judgeFailureCount === 0 && successCount >= 2,
  };
}

function initChart(pipelineIds) {
  const ctx = document.getElementById('score-chart').getContext('2d');
  if (state.chart) state.chart.destroy();

  state.chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: pipelineIds,
      datasets: [{
        label: 'Average Score',
        data: pipelineIds.map(() => 0),
        backgroundColor: pipelineIds.map(() => 'rgba(16, 185, 129, 0.6)'),
        borderColor: pipelineIds.map(() => 'rgba(16, 185, 129, 1)'),
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { color: '#888898' },
          grid: { color: '#2a2a3e' },
        },
        x: {
          ticks: { color: '#888898', maxRotation: 45 },
          grid: { display: false },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

function updateChart() {
  if (!state.chart) return;
  const labels = state.chart.data.labels;
  const avgs = labels.map((pipelineId) => {
    const scores = state.results
      .filter((entry) => entry.pipeline_id === pipelineId)
      .map((entry) => Number(entry.score || 0));
    return scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0;
  });

  const maxAvg = Math.max(...avgs, 0);
  state.chart.data.datasets[0].data = avgs;
  state.chart.data.datasets[0].backgroundColor = avgs.map((value) =>
    value === maxAvg && value > 0 ? 'rgba(16, 185, 129, 0.9)' : 'rgba(16, 185, 129, 0.4)'
  );
  state.chart.update('none');
}

function initTable(pipelineIds, taskIds) {
  const header = document.getElementById('table-header');
  header.innerHTML = '<th>Task</th>' + pipelineIds.map((pipeline) => `<th>${escapeHtml(pipeline)}</th>`).join('');

  const body = document.getElementById('table-body');
  body.innerHTML = taskIds.map((taskId) =>
    `<tr><td>${escapeHtml(taskId)}</td>${pipelineIds.map((pipeline) =>
      `<td id="cell-${taskId}-${pipeline}" class="score-cell" onclick="showDrilldown('${taskId}','${pipeline}')">-</td>`
    ).join('')}</tr>`
  ).join('');

  body.innerHTML += `<tr style="font-weight:700;border-top:2px solid var(--border)">
    <td>AVG</td>${pipelineIds.map((pipeline) =>
      `<td id="avg-${pipeline}" class="score-cell">-</td>`
    ).join('')}</tr>`;
}

function updateTableCell(data) {
  const cell = document.getElementById(`cell-${data.task_id}-${data.pipeline_id}`);
  if (cell) {
    cell.textContent = data.ok ? data.score : 'ERR';
    cell.className = 'score-cell ' + (
      !data.ok ? 'score-error' :
      data.score >= 70 ? 'score-high' :
      data.score >= 40 ? 'score-mid' : 'score-low'
    );
  }
  updateAverages();
}

function updateAverages() {
  const pipelineIds = [...new Set(state.results.map((entry) => entry.pipeline_id))];
  const allAvgs = {};

  pipelineIds.forEach((pipelineId) => {
    const scores = state.results
      .filter((entry) => entry.pipeline_id === pipelineId)
      .map((entry) => Number(entry.score || 0));
    const avgNumber = scores.length
      ? scores.reduce((sum, score) => sum + score, 0) / scores.length
      : null;
    allAvgs[pipelineId] = avgNumber || 0;
    const cell = document.getElementById(`avg-${pipelineId}`);
    if (cell) {
      cell.textContent = avgNumber == null ? '-' : avgNumber.toFixed(1);
      cell.className = 'score-cell ' + (
        avgNumber == null ? '' :
        avgNumber >= 70 ? 'score-high' :
        avgNumber >= 40 ? 'score-mid' : 'score-low'
      );
    }
  });

  const maxAvg = Math.max(...Object.values(allAvgs), 0);
  pipelineIds.forEach((pipelineId) => {
    const cell = document.getElementById(`avg-${pipelineId}`);
    if (cell) {
      cell.classList.toggle('best-score', allAvgs[pipelineId] === maxAvg && maxAvg > 0);
    }
  });
}

function updateThesisBanner(summary = currentSummary()) {
  const banner = document.getElementById('thesis-banner');
  if (!summary.valid_for_thesis) {
    banner.style.display = 'none';
    return;
  }

  const baseline = state.results.filter((entry) => entry.pipeline_id === 'single');
  const nonBaseline = state.results.filter((entry) => entry.pipeline_id !== 'single');
  if (!baseline.length || !nonBaseline.length) {
    banner.style.display = 'none';
    return;
  }

  const baseAvg = baseline.reduce((sum, entry) => sum + Number(entry.score || 0), 0) / baseline.length;
  const pipelineIds = [...new Set(nonBaseline.map((entry) => entry.pipeline_id))];
  let bestPipeline = '';
  let bestAvg = -1;
  pipelineIds.forEach((pipelineId) => {
    const scores = nonBaseline
      .filter((entry) => entry.pipeline_id === pipelineId)
      .map((entry) => Number(entry.score || 0));
    if (!scores.length) return;
    const avg = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    if (avg > bestAvg) {
      bestAvg = avg;
      bestPipeline = pipelineId;
    }
  });

  if (!bestPipeline || baseAvg <= 0) {
    banner.style.display = 'none';
    return;
  }

  const supported = bestAvg > baseAvg;
  const improvement = (((bestAvg - baseAvg) / baseAvg) * 100).toFixed(1);
  banner.style.display = 'block';
  banner.className = 'thesis-banner ' + (supported ? 'supported' : 'not-supported');
  banner.textContent = supported
    ? `THESIS SUPPORTED: "${bestPipeline}" beats baseline by ${improvement}% (${bestAvg.toFixed(2)} vs ${baseAvg.toFixed(2)})`
    : `THESIS NOT SUPPORTED: Baseline (${baseAvg.toFixed(2)}) matches or beats all synthesis pipelines (best: ${bestAvg.toFixed(2)})`;
}

function updateStats(summary = currentSummary()) {
  const container = document.getElementById('stats-row');
  const totalTime = state.results.reduce((sum, entry) => sum + Number(entry.wall_ms || 0), 0);
  const totalCalls = state.results.reduce((sum, entry) => sum + Number(entry.llm_calls || 0), 0);
  const avgAll = Number(summary.average_score_all_tasks || 0).toFixed(2);
  const avgSuccess = summary.average_score_success_only == null
    ? '-'
    : Number(summary.average_score_success_only).toFixed(2);

  container.innerHTML = `
    <div class="stat-card"><div class="stat-label">Success</div><div class="stat-value">${summary.success_count}/${summary.total_results}</div></div>
    <div class="stat-card"><div class="stat-label">Failures</div><div class="stat-value">${summary.failure_count}</div></div>
    <div class="stat-card"><div class="stat-label">Judge Failures</div><div class="stat-value">${summary.judge_failure_count || 0}</div></div>
    <div class="stat-card"><div class="stat-label">Avg All Tasks</div><div class="stat-value">${avgAll}</div></div>
    <div class="stat-card"><div class="stat-label">Avg Success Only</div><div class="stat-value">${avgSuccess}</div></div>
    <div class="stat-card"><div class="stat-label">Total Time</div><div class="stat-value">${(totalTime / 1000).toFixed(1)}s</div></div>
    <div class="stat-card"><div class="stat-label">LLM Calls</div><div class="stat-value">${totalCalls}</div></div>
  `;
}

async function showDrilldown(taskId, pipelineId) {
  const detail = await ensureRunDetails(state.currentRunId);
  if (!detail) return;

  const result = (detail.results || []).find(
    (entry) => entry.task_id === taskId && entry.pipeline_id === pipelineId
  );
  if (!result) return;

  const task = (detail.config?.tasks || state.tasks || []).find((entry) => entry.id === taskId) || {
    id: taskId,
    prompt: result.task_prompt,
    rubric: result.task_rubric,
  };

  document.getElementById('modal-title').textContent = `${pipelineId} - ${taskId}`;

  let html = '';
  if (task) {
    html += `<div class="modal-section">
      <div class="modal-label">Task</div>
      <div class="modal-text">${escapeHtml(task.prompt || '')}</div>
    </div>
    <div class="modal-section">
      <div class="modal-label">Rubric</div>
      <div class="modal-text">${escapeHtml(task.rubric || '')}</div>
    </div>`;
  }

  html += `<div class="modal-section">
    <div class="modal-label">Score</div>
    <div class="stat-value ${result.score >= 70 ? 'score-high' : result.score >= 40 ? 'score-mid' : 'score-low'}">${Number(result.score || 0)}/100</div>
  </div>
  <div class="modal-section">
    <div class="modal-label">Judge Reasoning</div>
    <div class="modal-text">${escapeHtml(result.judge_reasoning || 'N/A')}</div>
  </div>
  <div class="modal-section">
    <div class="modal-label">Judge Metadata</div>
    <div class="modal-text">backend=${escapeHtml(result.judge_backend || '')} model=${escapeHtml(result.judge_model || '')} judge_ok=${String(result.judge_ok !== false)}</div>
  </div>`;

  if (!result.ok) {
    html += `<div class="modal-section">
      <div class="modal-label">Pipeline Error</div>
      <div class="modal-text" style="color:var(--danger)">${escapeHtml(result.error || 'Unknown error')}</div>
    </div>`;
  }
  if (result.judge_ok === false) {
    html += `<div class="modal-section">
      <div class="modal-label">Judge Error</div>
      <div class="modal-text" style="color:var(--warning)">${escapeHtml(result.judge_error || 'Unknown judge error')}</div>
    </div>`;
  }

  html += `<div class="modal-section">
    <div class="modal-label">Metrics</div>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
      <span>Wall time: ${(Number(result.wall_ms || 0) / 1000).toFixed(1)}s</span>
      <span>LLM calls: ${Number(result.llm_calls || 0)}</span>
      <span>Tokens: ${Number(result.total_tokens || result.tokens || 0)}</span>
    </div>
  </div>`;

  if (result.final_text) {
    html += `<div class="modal-section">
      <div class="modal-label">Full Output</div>
      <div class="modal-text">${escapeHtml(result.final_text)}</div>
    </div>`;
  }

  if (result.steps && result.steps.length) {
    html += `<div class="modal-section">
      <div class="modal-label">Step Traces</div>
      <div class="step-trace step-header">
        <span>Phase</span><span>Model</span><span>Latency</span><span>Tokens</span><span>Preview</span>
      </div>`;
    result.steps.forEach((step) => {
      html += `<div class="step-trace">
        <span>${escapeHtml(step.phase || '')}</span>
        <span>${escapeHtml(step.model || '')}</span>
        <span>${(Number(step.latency_ms || 0) / 1000).toFixed(1)}s</span>
        <span>${Number(step.tokens || 0)}</span>
        <span style="color:var(--text-dim)">${escapeHtml(truncate(step.text_preview || '', 80))}</span>
      </div>`;
    });
    html += '</div>';
  }

  document.getElementById('modal-body').innerHTML = html;
  document.getElementById('modal').style.display = 'flex';
}

async function ensureRunDetails(runId) {
  if (!runId) return null;
  if (state.currentRunDetails && state.currentRunDetails.run_id === runId) {
    return state.currentRunDetails;
  }
  try {
    const resp = await fetch(`/api/results/${runId}`);
    const data = await resp.json();
    if (!resp.ok || data.error) return null;
    state.currentRunDetails = data;
    return data;
  } catch (e) {
    console.error('Failed to fetch run details:', e);
    return null;
  }
}

function closeModal() {
  document.getElementById('modal').style.display = 'none';
}

function exportResults(format) {
  if (!state.currentRunId) return;
  window.open(`/api/export/${state.currentRunId}?format=${format}`, '_blank');
}

async function loadRun(runId) {
  try {
    const resp = await fetch(`/api/results/${runId}`);
    const data = await resp.json();
    if (!resp.ok || !data || data.error) {
      alert('Could not load run');
      return;
    }

    state.currentRunId = runId;
    state.currentRunDetails = data;
    state.results = data.results || [];
    if (Array.isArray(data.config?.tasks) && data.config.tasks.length) {
      state.tasks = data.config.tasks;
    }

    const pipelineIds = [...new Set(state.results.map((entry) => entry.pipeline_id))];
    const taskIds = [...new Set(state.results.map((entry) => entry.task_id))];

    document.getElementById('results-panel').style.display = 'block';
    document.getElementById('result-run-id').textContent = `(${runId})`;
    document.getElementById('progress-text').textContent = `Loaded from history: ${data.status || 'completed'}`;
    document.getElementById('progress-bar').style.width = '100%';
    updateActiveRunBanner();

    initChart(pipelineIds);
    initTable(pipelineIds, taskIds);
    state.results.forEach((entry) => updateTableCell(entry));
    updateChart();
    updateThesisBanner(data.summary || currentSummary());
    updateStats(data.summary || currentSummary());
    document.getElementById('btn-rerun-failed').disabled = !((data.summary || {}).failure_count > 0);
  } catch (e) {
    console.error('Failed to load run:', e);
  }
}

async function rerunFailed() {
  if (!state.currentRunId) return;
  try {
    const sourceResults = (state.currentRunDetails?.results || state.results || [])
      .filter((entry) => entry.ok === false);
    const pipelineIds = [...new Set(sourceResults.map((entry) => entry.pipeline_id))];
    const taskIds = [...new Set(sourceResults.map((entry) => entry.task_id))];

    const resp = await fetch(`/api/rerun-failed/${state.currentRunId}`, { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      const details = (data.details || []).join('\n');
      alert(details ? `${data.error}\n\n${details}` : (data.error || `HTTP ${resp.status}`));
      return;
    }

    state.currentRunId = data.run_id;
    state.currentRunDetails = {
      run_id: data.run_id,
      results: [],
      summary: {},
    };
    state.results = [];
    if (pipelineIds.length && taskIds.length) {
      initChart(pipelineIds);
      initTable(pipelineIds, taskIds);
    }
    document.getElementById('result-run-id').textContent = `(${data.run_id})`;
    document.getElementById('progress-text').textContent = `0/${data.total}`;
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('btn-rerun-failed').disabled = true;
    connectSSE(data.run_id, data.total);
  } catch (e) {
    console.error('Failed to rerun failed cells:', e);
    alert(`Failed to rerun failed cells: ${e.message}`);
  }
}

function truncate(text, limit) {
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}
