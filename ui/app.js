/**
 * Lexplain – Premium UI JavaScript
 * Handles: SSE streaming, history, tab switching, result rendering, feedback
 */

'use strict';

// ── State ────────────────────────────────────────────────────────────────────
let currentCaseId = null;
let currentTenantId = 'ui_tenant';
let currentResults = null;
let historyData = [];

// Agent steps (shown in stepper, in order)
const AGENT_STEPS = [
  'Agent 0 — Ingestion',
  'Agent 1 — Normalization',
  'Agent 2 — Segmentation',
  'Agent 3 — Role Tagging',
  'Agent 4A — Entity Extraction',
  'Agent 4B — Event Builder',
  'Agent 5A — Legal Fact Normalizer',
  'Agent 5B — Legal Signal Extractor',
  'Agent 5C — Statute Retriever',
  'Agent 6 — Ingredient Evaluator',
  'Agent 7 — Precedent Comparator',
  'Agent 8 — Loophole Miner',
  'Agent 9 — Final Argument Engine',
  'Agent 5D — Feedback Memory',
];

// ── DOM Refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildStepper();
  loadHistory();
  setupTabNav();
  setupAutoResize();
  setupSimplifiedToggle();

  $('analyze-btn').addEventListener('click', startAnalysis);
  $('btn-new-analysis').addEventListener('click', resetToNew);
  $('btn-open-feedback').addEventListener('click', openFeedbackDrawer);
  $('attach-btn').addEventListener('click', () => $('file-upload').click());
  $('file-upload').addEventListener('change', handleFileUpload);

  // Auto-submit on Ctrl+Enter
  $('case-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) startAnalysis();
  });
});

// ── Auto-resize textarea ──────────────────────────────────────────────────────
function setupAutoResize() {
  const ta = $('case-input');
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
  });
}

// ── Tab Navigation ────────────────────────────────────────────────────────────
function setupTabNav() {
  $$('.nav-item[data-tab]').forEach(item => {
    item.addEventListener('click', () => switchTab(item.dataset.tab));
  });
}

function switchTab(tab) {
  $$('.nav-item[data-tab]').forEach(n => n.classList.toggle('active', n.dataset.tab === tab));

  // hide all views
  ['analyze', 'documents', 'saved', 'history'].forEach(v =>
    $(`view-${v}`).style.display = 'none'
  );
  $(`view-${tab}`).style.display = '';

  // sidebar history list visibility
  $('history-list').style.display = (tab === 'history') ? 'block' : 'none';

  const titles = {
    analyze: 'Legal Analysis Workspace',
    history: 'History <span>— past analyses</span>',
    documents: 'Documents <span>— case files</span>',
    saved: 'Saved <span>— bookmarked analyses</span>',
  };
  $('topbar-title').innerHTML = titles[tab] || tab;
}

// ── Stepper Build ─────────────────────────────────────────────────────────────
function buildStepper() {
  const container = $('agent-stepper');
  container.innerHTML = AGENT_STEPS.map((name, i) => `
    <div class="agent-step" id="step-${i}">
      <div class="step-dot">
        <span>${i + 1}</span>
        <div class="step-pulse"></div>
      </div>
      <div class="step-info">
        <div class="step-name">${name}</div>
        <div class="step-detail" id="step-detail-${i}">Waiting…</div>
      </div>
    </div>
  `).join('');
}

function resetStepper() {
  AGENT_STEPS.forEach((_, i) => {
    const el = $(`step-${i}`);
    el.className = 'agent-step';
    const dot = el.querySelector('.step-dot');
    dot.innerHTML = `<span>${i + 1}</span><div class="step-pulse"></div>`;
    $(`step-detail-${i}`).textContent = 'Waiting…';
  });
}

function stepRunning(agentName, detail) {
  const i = AGENT_STEPS.findIndex(n => agentName.startsWith(n.split(' — ')[0]));
  if (i < 0) return;
  const el = $(`step-${i}`);
  el.className = 'agent-step running';
  $(`step-detail-${i}`).textContent = detail;
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  $('current-stage-label').textContent = agentName;
}

function stepDone(agentName, detail) {
  const i = AGENT_STEPS.findIndex(n => agentName.startsWith(n.split(' — ')[0]));
  if (i < 0) return;
  const el = $(`step-${i}`);
  el.className = 'agent-step done';
  const dot = el.querySelector('.step-dot');
  dot.innerHTML = '<span>✔</span>';
  $(`step-detail-${i}`).textContent = detail;
}

function stepError(agentName, detail) {
  const i = AGENT_STEPS.findIndex(n => agentName.startsWith(n.split(' — ')[0]));
  if (i < 0) return;
  const el = $(`step-${i}`);
  el.className = 'agent-step error';
  $(`step-detail-${i}`).textContent = '⚠ ' + detail;
}

// ── Start Analysis ────────────────────────────────────────────────────────────
async function startAnalysis() {
  const text = $('case-input').value.trim();
  if (!text) { showToast('Please enter a case description first.'); return; }

  // Switch to analyze tab if not there
  switchTab('analyze');

  // UI: enter progress mode
  $('analyze-btn').disabled = true;
  $('analyze-btn').textContent = 'Analyzing...';
  $('empty-state').style.display = 'none';
  $('results-panel').classList.remove('visible');
  $('feedback-section').classList.remove('visible');

  resetStepper();

  const progressPanel = $('progress-panel');
  progressPanel.classList.add('open');
  progressPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  $('pipeline-spinner').classList.remove('hidden');
  $('current-stage-label').textContent = 'Initialising…';

  const role = $('role-select').value;
  const tenantId = currentTenantId;

  try {
    // 1. POST to start analysis
    const resp = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, role, tenant_id: tenantId }),
    });
    const { job_id, error } = await resp.json();
    if (!job_id) { throw new Error(error || 'Failed to start analysis'); }

    // 2. Listen to SSE stream
    await listenSSE(job_id);

  } catch (err) {
    showToast('Error: ' + err.message);
    $('analyze-btn').disabled = false;
    $('analyze-btn').textContent = 'Analyze';
    progressPanel.classList.remove('open');
  }
}

// ── SSE Listener ──────────────────────────────────────────────────────────────
function listenSSE(jobId) {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/stream/${jobId}`);

    es.onmessage = async (e) => {
      let event;
      try { event = JSON.parse(e.data); } catch { return; }

      if (event.type === 'progress') {
        if (event.status === 'running') {
          stepRunning(event.stage, event.detail || 'Processing…');
        } else if (event.status === 'done') {
          stepDone(event.stage, event.detail || 'Done');
        } else if (event.status === 'error') {
          stepError(event.stage, event.detail || 'Error');
        }
      } else if (event.type === 'done') {
        es.close();
        currentCaseId = event.case_id;
        currentTenantId = event.tenant_id || 'ui_tenant';

        // Mark all remaining as done
        $('pipeline-spinner').classList.add('hidden');
        $('current-stage-label').textContent = '✅ Analysis complete';

        // Load results and render
        await loadAndRenderCase(currentTenantId, currentCaseId);

        // Collapse progress, show results
        setTimeout(() => {
          $('progress-panel').classList.remove('open');
          setTimeout(() => {
            $('results-panel').classList.add('visible');
            $('feedback-section').classList.add('visible');
            $('results-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
          }, 350);
        }, 800);

        $('analyze-btn').disabled = false;
        $('analyze-btn').textContent = 'Analyze';

        // Refresh history badge
        loadHistory();

        resolve();
      } else if (event.type === 'error') {
        es.close();
        showToast('Pipeline error: ' + (event.message || 'Unknown error'));
        $('analyze-btn').disabled = false;
        $('analyze-btn').textContent = 'Analyze';
        $('pipeline-spinner').classList.add('hidden');
        reject(new Error(event.message));
      }
    };
    es.onerror = () => {
      es.close();
      reject(new Error('SSE connection lost'));
    };
  });
}

// ── Load & Render Case ────────────────────────────────────────────────────────
async function loadAndRenderCase(tenantId, caseId) {
  const resp = await fetch(`/api/case/${tenantId}/${caseId}`);
  if (!resp.ok) return;
  currentResults = await resp.json();
  renderResults(currentResults);
}

function renderResults(data) {
  const fa = data.final_argument || {};
  const pc = data.precedent_comparison || {};
  const lh = data.loopholes || {};

  // ── Summary
  const summary = fa.case_summary || fa.summary || 'No summary available.';
  $('summary-text').innerHTML = summary
    .replace(/IPC \d+/g, s => `<strong>${s}</strong>`)
    .replace(/Section \d+/g, s => `<strong>${s}</strong>`);

  // ── Structured Loopholes
  const structuredLoopholes = fa.structured_loophole_arguments || [];
  const structCard = $('card-structured-loopholes');
  const structList = $('structured-loopholes-list');

  if (structuredLoopholes.length > 0) {
    structCard.style.display = '';
    structList.innerHTML = structuredLoopholes.map(lh => `
      <div class="reasoning-item" style="margin-bottom:12px;">
        <span class="reasoning-bullet">▸</span>
        <span><strong>${escHtml(lh.title || 'Gap Identified')}:</strong> ${escHtml(lh.argument || '')}</span>
      </div>
    `).join('');
  } else {
    structCard.style.display = 'none';
  }

  // ── Final Argument
  const finalArgumentText = fa.final_argument_text || (typeof fa.final_argument === 'string' ? fa.final_argument : '');
  const finalArgCard = $('card-final-argument');

  if (finalArgumentText) {
    if (finalArgCard) finalArgCard.style.display = '';

    // Split by newlines and wrap in paragraphs, handling headers
    const formattedHtml = finalArgumentText
      .split('\n')
      .map(line => line.trim())
      .filter(line => line !== '')
      .map(line => {
        if (line.match(/^\d+\.\s/)) return `<h4 style="margin-top:16px; margin-bottom:8px; color:var(--text);">${escHtml(line)}</h4>`;
        if (line.match(/^=/)) return `<hr style="margin: 16px 0; border-top: 1px solid var(--border);"/>`;
        if (line.match(/^-{10,}/)) return '';
        return `<p style="margin-bottom:8px;">${escHtml(line)}</p>`;
      })
      .join('');

    $('final-argument-text').innerHTML = formattedHtml;
  } else {
    if (finalArgCard) finalArgCard.style.display = 'none';
  }

  // ── Laws
  const sections = fa.applicable_sections || [];
  const lawsGrid = $('laws-grid');
  if (sections.length) {
    lawsGrid.innerHTML = sections.map(s => {
      const badge = s.status === 'valid' ? 'badge-valid'
        : s.status === 'weak' ? 'badge-weak'
          : 'badge-invalid';
      const label = s.status === 'valid' ? 'Valid'
        : s.status === 'weak' ? 'Weak'
          : 'Not Satisfied';
      const score = s.overall_score !== undefined
        ? `<span class="law-score">${(s.overall_score * 100).toFixed(0)}%</span>` : '';
      return `
        <div class="law-item">
          <div class="law-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 13.5V16.5l-4-4 4-4v3a8 8 0 1 1-6.5-12.8" stroke="none"/><path d="M14 13.5V20"/><path d="M11.5 6.5l2-2 4 4-2 2-4-4z"/><path d="M18 10l-4-4"/></svg>
          </div>
          <span class="law-text">Section ${s.section} IPC — ${escHtml(s.heading || s.section_heading || '')}</span>
          <span class="law-badge ${badge}">${label}</span>
          ${score}
        </div>`;
    }).join('');
  } else {
    lawsGrid.innerHTML = '<p style="color:var(--text-3);font-size:13px;">No sections retrieved.</p>';
  }

  // ── Precedents
  const patterns = (pc.matched_patterns || []).slice(0, 6);
  const refs = pc.precedent_references || [];
  const precedentsGrid = $('precedents-grid');
  if (patterns.length) {
    precedentsGrid.innerHTML = patterns.map((p, i) => {
      const ref = refs[i] || {};
      const caseName = (ref.case_name && ref.case_name !== 'Unknown Case')
        ? ref.case_name
        : (ref.source ? ref.source.replace(/_EN\.json$/, '').replace(/_/g, ' ') : 'Unnamed Case');
      return `
        <div class="precedent-card">
          <div class="precedent-title">${escHtml(caseName)}</div>
          <div class="precedent-source">${ref.citation || ref.source || ''}</div>
          <div class="precedent-snippet">${escHtml(p.why_matched || p.pattern || '')}</div>
        </div>`;
    }).join('');
  } else {
    precedentsGrid.innerHTML = '<p style="color:var(--text-3);font-size:13px;">No precedents matched.</p>';
  }

  // ── Reasoning
  const reasoning = fa.ingredient_analysis || fa.legal_reasoning || [];
  const reasoningList = $('reasoning-list');
  if (Array.isArray(reasoning) && reasoning.length) {
    reasoningList.innerHTML = reasoning.map((r, i) => {
      const text = typeof r === 'string' ? r : (r.reasoning || r.analysis || JSON.stringify(r));
      return `
        <div class="reasoning-item">
          <span class="reasoning-bullet">▸</span>
          <span><strong>${i + 1}.</strong> ${escHtml(text)}</span>
        </div>`;
    }).join('');
  } else if (typeof reasoning === 'object' && reasoning !== null) {
    let idx = 1;
    const items = Object.entries(reasoning).map(([k, v]) =>
      `<div class="reasoning-item"><span class="reasoning-bullet">▸</span><span><strong>${idx++}. ${escHtml(k)}</strong>: ${escHtml(String(v))}</span></div>`
    ).join('');
    reasoningList.innerHTML = items || '<p style="color:var(--text-3);font-size:13px;">No reasoning data.</p>';
  } else {
    reasoningList.innerHTML = '<p style="color:var(--text-3);font-size:13px;">No reasoning data.</p>';
  }

  // ── Loopholes
  const loops = lh.loopholes || [];
  const loopholeList = $('loophole-list');
  if (loops.length) {
    loopholeList.innerHTML = loops.map(l => {
      const text = typeof l === 'string' ? l
        : (l.description || l.issue || l.loophole || JSON.stringify(l));
      const ingredient = l.ingredient ? `<strong>${l.ingredient}</strong>: ` : '';
      return `<div class="loophole-item">${ingredient}${escHtml(text)}</div>`;
    }).join('');
  } else {
    loopholeList.innerHTML = '<div class="loophole-item" style="opacity:0.6;">No critical loopholes identified.</div>';
  }

  // ── Citations
  window.currentCitations = refs.filter(r => r.case_name || r.source);
  renderCitationsPage(0);
}

function renderCitationsPage(pageIndex) {
  const citations = window.currentCitations || [];
  const citationsList = $('citations-list');
  const dotsContainer = $('citations-dots');

  if (!citations.length) {
    if (citationsList) citationsList.innerHTML = '<p style="color:var(--text-3);font-size:13px;">No citations available.</p>';
    if (dotsContainer) dotsContainer.innerHTML = '';
    return;
  }

  const ITEMS_PER_PAGE = 2;
  const totalPages = Math.ceil(citations.length / ITEMS_PER_PAGE);
  const safePage = Math.max(0, Math.min(pageIndex, totalPages - 1));
  window.currentCitationPage = safePage;

  const start = safePage * ITEMS_PER_PAGE;
  const pageItems = citations.slice(start, start + ITEMS_PER_PAGE);

  if (citationsList) {
    citationsList.innerHTML = pageItems.map(c => {
      const name = (c.case_name && c.case_name !== 'Unknown Case') ? c.case_name : (c.source || 'Unnamed Case');
      return `
      <div class="citation-item">
        <span class="citation-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path></svg>
        </span>
        <span class="citation-name">${escHtml(name)} <span style="color:var(--text-3);font-size:12px;margin-left:6px;">${escHtml(c.citation || '')}</span></span>
        <a href="#" class="read-more" onclick="return false;">[Read More]</a>
      </div>`;
    }).join('');
  }

  if (dotsContainer) {
    if (totalPages > 1) {
      let dotsHtml = '';
      for (let i = 0; i < totalPages; i++) {
        dotsHtml += `<span class="carousel-dot" onclick="renderCitationsPage(${i})" style="display:inline-block; width:8px; height:8px; margin:0 4px; border-radius:50%; background-color:${i === safePage ? 'var(--primary)' : 'var(--border-strong)'}; cursor:pointer; transition:background-color 0.2s;"></span>`;
      }
      dotsContainer.innerHTML = dotsHtml;
    } else {
      dotsContainer.innerHTML = '';
    }
  }
}

// ── History ───────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const resp = await fetch('/api/history');
    historyData = await resp.json();
    renderHistorySidebar();
    $('history-badge').textContent = historyData.length;
  } catch {
    historyData = [];
  }
}

function renderHistorySidebar() {
  const list = $('history-list');
  if (!historyData.length) {
    list.innerHTML = '<div style="padding:16px 20px;font-size:12px;color:var(--text-3);">No history yet.</div>';
    return;
  }
  list.innerHTML = historyData.map((h, i) => {
    const date = new Date(h.timestamp).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
    const timeStr = new Date(h.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
    const snippet = (h.snippet || '').substring(0, 60) + '…';
    const chips = (h.valid_sections || []).slice(0, 3).map(s =>
      `<span class="section-chip">IPC ${s}</span>`
    ).join('');
    return `
      <div class="history-item" onclick="loadHistoryCase('${h.tenant_id}','${h.case_id}', ${i})">
        <div class="history-title">${escHtml(snippet)}</div>
        <div class="history-meta">${date} · ${timeStr}</div>
        <div class="history-sections">${chips}</div>
      </div>`;
  }).join('');
}

async function loadHistoryCase(tenantId, caseId, idx) {
  // Highlight in sidebar
  $$('.history-item').forEach((el, i) => el.classList.toggle('active', i === idx));

  // Switch to analyze tab
  switchTab('analyze');

  $('empty-state').style.display = 'none';
  $('progress-panel').classList.remove('open');
  $('results-panel').classList.remove('visible');
  $('feedback-section').classList.remove('visible');

  try {
    currentCaseId = caseId;
    currentTenantId = tenantId;
    await loadAndRenderCase(tenantId, caseId);

    $('results-panel').classList.add('visible');
    loadExistingFeedback(tenantId, caseId);
    setTimeout(() => $('feedback-section').classList.add('visible'), 200);
    showToast('Loaded case from history');
  } catch (e) {
    showToast('Could not load case: ' + e.message);
  }
}

// ── Feedback ──────────────────────────────────────────────────────────────────
function openFeedbackDrawer() {
  const drawer = $('feedback-drawer');
  const isOpen = drawer.classList.contains('open');
  if (isOpen) {
    drawer.classList.remove('open');
    $('btn-open-feedback').textContent = '📝 Rate this Analysis (Human Review)';
  } else {
    drawer.classList.add('open');
    $('btn-open-feedback').textContent = '▲ Close Feedback';
  }
}

function toggleYN(btn) {
  const comp = btn.closest('.feedback-component');
  const isYes = btn.classList.contains('yes');
  const isNo = btn.classList.contains('no');

  // Update button states
  comp.querySelectorAll('.yn-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');

  // Show / hide comment box
  const comment = comp.querySelector('.fc-comment');
  if (isNo) {
    comment.classList.add('visible');
    comment.placeholder = 'What was incorrect or missing?';
  } else if (isYes) {
    comment.classList.add('visible');
    comment.placeholder = 'What made this correct? (optional)';
  } else {
    comment.classList.remove('visible');
  }
}

async function submitFeedback() {
  if (!currentCaseId) { showToast('No active case to rate.'); return; }

  const components = {};
  $$('.feedback-component').forEach(comp => {
    const key = comp.dataset.component;
    const yes = comp.querySelector('.yn-btn.yes.selected');
    const no = comp.querySelector('.yn-btn.no.selected');
    const text = comp.querySelector('.fc-comment').value.trim();
    if (yes || no) {
      components[key] = { correct: !!yes, comment: text };
    }
  });

  if (!Object.keys(components).length) {
    showToast('Rate at least one component before submitting.');
    return;
  }

  $('btn-submit-fb').disabled = true;
  $('btn-submit-fb').textContent = 'Saving…';

  try {
    const resp = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        case_id: currentCaseId,
        tenant_id: currentTenantId,
        components,
      }),
    });
    const data = await resp.json();
    $('feedback-form').style.display = 'none';
    $('feedback-submitted').classList.add('visible');
    showToast(`✅ Feedback saved! ${data.patterns_learned} pattern(s) learned.`);
  } catch (e) {
    showToast('Failed to save feedback: ' + e.message);
    $('btn-submit-fb').disabled = false;
    $('btn-submit-fb').textContent = 'Submit Feedback & Learn →';
  }
}

async function loadExistingFeedback(tenantId, caseId) {
  try {
    const resp = await fetch(`/api/feedback/${tenantId}/${caseId}`);
    const fb = await resp.json();
    if (fb && fb.components && Object.keys(fb.components).length) {
      // Pre-fill form
      Object.entries(fb.components).forEach(([key, val]) => {
        const comp = document.querySelector(`.feedback-component[data-component="${key}"]`);
        if (!comp) return;
        const btnClass = val.correct ? '.yn-btn.yes' : '.yn-btn.no';
        const btn = comp.querySelector(btnClass);
        if (btn) {
          toggleYN(btn);
          if (val.comment) comp.querySelector('.fc-comment').value = val.comment;
        }
      });
    }
  } catch { /* no prior feedback */ }
}

// ── Card Toggle ───────────────────────────────────────────────────────────────
function toggleCard(cardId) {
  const body = $('body-' + cardId);
  const expand = $('expand-' + cardId);
  const isOpen = !body.classList.contains('collapsed');
  body.classList.toggle('collapsed', isOpen);
  expand.classList.toggle('open', !isOpen);
}

// ── Reset to new analysis ─────────────────────────────────────────────────────
function resetToNew() {
  switchTab('analyze');
  $('case-input').value = '';
  $('case-input').style.height = 'auto';
  $('results-panel').classList.remove('visible');
  $('feedback-section').classList.remove('visible');
  $('progress-panel').classList.remove('open');
  $('empty-state').style.display = '';
  $('feedback-drawer').classList.remove('open');
  $('feedback-form').style.display = '';
  $('feedback-submitted').classList.remove('visible');
  $('btn-submit-fb').disabled = false;
  $('btn-submit-fb').textContent = 'Submit Feedback & Learn →';
  $$('.yn-btn').forEach(b => b.classList.remove('selected'));
  $$('.fc-comment').forEach(t => { t.value = ''; t.classList.remove('visible'); });
  currentCaseId = null;
  currentResults = null;
  resetStepper();
}

// ── File Upload ───────────────────────────────────────────────────────────────
function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    $('case-input').value = ev.target.result;
    $('case-input').dispatchEvent(new Event('input'));
    showToast(`Loaded: ${file.name}`);
  };
  reader.readAsText(file);
  e.target.value = '';
}

// ── Simplified toggle ─────────────────────────────────────────────────────────
function setupSimplifiedToggle() {
  const toggle = $('simplified-toggle');
  if (!toggle) return;
  toggle.addEventListener('click', () => {
    toggle.classList.toggle('on');
    const on = toggle.classList.contains('on');
    // In simplified mode, hide low-level cards
    ['card-reasoning', 'card-citations'].forEach(id => {
      const el = $(id);
      if (el) el.style.display = on ? 'none' : '';
    });
    showToast(on ? 'Simplified view on' : 'Full view on');
  });
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let _toastTimer = null;
function showToast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove('show'), 3200);
}

// ── Escape HTML ───────────────────────────────────────────────────────────────
function escapeHTML(str) {
  if (typeof str !== 'string') return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escHtml(str) {
  const noAsterisks = String(str).replace(/\*/g, '');
  return escapeHTML(noAsterisks);
}
