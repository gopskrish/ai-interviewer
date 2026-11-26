// The session page: polls /api/sessions/{id} every 2 s and shows status, transcript and scorecard.
// practice.js reuses the exported helpers.

const FINISHED = new Set(['scoring', 'done', 'failed', 'ended']);

export function isFinished(data) {
  return !data || FINISHED.has(data.status);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function renderTranscript(list, turns) {
  const wasAtBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
  list.replaceChildren(...turns.map((turn) => {
    const item = el('li', turn.speaker === 'ai' ? 'ai' : 'candidate');
    const who = turn.speaker === 'ai' ? 'AI' : 'Candidate';
    item.append(el('span', 'who', `[${turn.at}] ${who}: `), el('span', 'text', turn.text));
    if (turn.interrupted) item.append(el('span', 'note', ' (interrupted)'));
    return item;
  }));
  if (wasAtBottom) list.scrollTop = list.scrollHeight;
}

export function renderScorecard(box, card) {
  if (!card) return;
  const parts = [];
  for (const topic of card.topics) {
    const section = el('section', 'topic');
    const label = topic.not_enough_signal || topic.score === null ? 'Not enough signal' : `${topic.score}/4`;
    section.append(el('h3', '', `${topic.name}: ${label}`));
    const lists = [['Strengths', topic.strengths], ['Concerns', topic.concerns]];
    for (const [title, items] of lists) {
      if (!items || !items.length) continue;
      section.append(el('p', 'label', title));
      const ul = el('ul');
      for (const item of items) ul.append(el('li', '', item));
      section.append(ul);
    }
    if (topic.quotes && topic.quotes.length) {
      section.append(el('p', 'label', 'Evidence'));
      const ul = el('ul', 'quotes');
      for (const quote of topic.quotes) ul.append(el('li', '', `[${quote.at}] "${quote.text}"`));
      section.append(ul);
    } else if (topic.evidence_missing) {
      section.append(el('p', 'warning', 'No verified quotes for this score. Check the transcript.'));
    }
    parts.push(section);
  }
  const facts = el('p', 'hint', `Ended: ${card.end_reason || 'unknown'} · Flags: ${card.flags.length ? card.flags.join(', ') : 'none'}`);
  const note = el('p', 'hint', 'Evidence for a human reviewer. Read the transcript before deciding anything.');
  box.replaceChildren(note, ...parts, facts);
}

export function poll(sessionId, onData, intervalMs = 2000) {
  let stopped = false;
  async function once() {
    try {
      const response = await fetch(`/api/sessions/${sessionId}`, { cache: 'no-store' });
      if (response.ok) onData(await response.json());
      else if (response.status === 404) onData(null);
    } catch {
      // The service may be restarting; keep trying.
    }
    if (!stopped) setTimeout(once, intervalMs);
  }
  once();
  return () => { stopped = true; };
}

export async function post(url) {
  const response = await fetch(url, { method: 'POST' });
  let body = {};
  try { body = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) throw new Error(body.error || `The request failed (${response.status}).`);
  return body;
}

function setupSessionPage(root) {
  const id = root.dataset.sessionId;
  const $ = (name) => document.getElementById(name);
  const beginButton = $('begin');
  const endButton = $('end');
  const showError = (message) => { $('error').textContent = message || ''; $('error').hidden = !message; };

  beginButton.addEventListener('click', async () => {
    beginButton.disabled = true;
    try { await post(`/api/sessions/${id}/begin`); } catch (error) { showError(error.message); beginButton.disabled = false; }
  });
  endButton.addEventListener('click', async () => {
    if (!confirm('End the interview now? The AI says goodbye and the bot leaves the call.')) return;
    endButton.disabled = true;
    endButton.textContent = 'Ending…';
    try { await post(`/api/sessions/${id}/end`); } catch (error) { showError(error.message); }
    endButton.textContent = 'End interview';
  });

  poll(id, (data) => {
    if (!data) { showError('This session no longer exists.'); return; }
    $('status').textContent = data.status_label;
    $('bot').textContent = data.bot_label || '—';
    $('speaking').textContent = data.speaking || '—';
    showError(data.error);
    beginButton.disabled = !(data.live && data.connected && !data.begun);
    endButton.disabled = !data.live;
    $('dl-transcript').hidden = !data.downloads.transcript;
    $('dl-scorecard').hidden = !data.downloads.scorecard;
    renderTranscript($('transcript'), data.transcript);
    renderScorecard($('scorecard'), data.scorecard);
  });
}

const sessionRoot = document.getElementById('session');
if (sessionRoot) setupSessionPage(sessionRoot);
