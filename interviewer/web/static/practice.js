// The practice page: the browser microphone stands in for the Teams call.
// It speaks the same websocket protocol as Attendee: realtime_audio.mixed in
// (PCM16, 16 kHz, every ~100 ms) and realtime_audio.bot_output back.

import { base64ToInt16, downsampleTo16k, int16ToBase64 } from './audio-utils.mjs';
import { isFinished, poll, post, renderTranscript } from './session.js';

const root = document.getElementById('practice');
const id = root.dataset.sessionId;
const $ = (name) => document.getElementById(name);
const startButton = $('start');
const beginButton = $('begin');
const endButton = $('end');
const setStatus = (text) => { $('status').textContent = text; };

let ctx = null;
let stream = null;
let ws = null;
let playHead = 0;
let queued = [];
let queuedLength = 0;
let stopped = false;
let t0 = 0;

function sendAudio(float32) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const pcm = downsampleTo16k(float32, ctx.sampleRate);
  ws.send(JSON.stringify({
    trigger: 'realtime_audio.mixed',
    bot_id: 'practice',
    data: { chunk: int16ToBase64(pcm), timestamp_ms: Math.round(performance.now() - t0), sample_rate: 16000 },
  }));
}

function onFrame(frame, block) {
  queued.push(frame);
  queuedLength += frame.length;
  if (queuedLength < block) return;
  const all = new Float32Array(queuedLength);
  let offset = 0;
  for (const part of queued) { all.set(part, offset); offset += part.length; }
  const usable = queuedLength - (queuedLength % block);
  sendAudio(all.subarray(0, usable));
  const rest = all.slice(usable);
  queued = rest.length ? [rest] : [];
  queuedLength = rest.length;
}

function play(message) {
  if (message.trigger !== 'realtime_audio.bot_output' || !ctx) return;
  const pcm = base64ToInt16(message.data.chunk);
  if (!pcm.length) return;
  const buffer = ctx.createBuffer(1, pcm.length, message.data.sample_rate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  const startAt = Math.max(ctx.currentTime + 0.05, playHead);
  source.start(startAt);
  playHead = startAt + buffer.duration;
}

function stopAll() {
  stopped = true;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (ws && ws.readyState <= WebSocket.OPEN) ws.close(1000);
  startButton.disabled = true;
  beginButton.disabled = true;
  endButton.disabled = true;
}

function stopAfterPlayback() {
  const remainingMs = ctx ? Math.max(0, (playHead - ctx.currentTime) * 1000) : 0;
  setTimeout(stopAll, remainingMs + 300);
}

startButton.addEventListener('click', async () => {
  startButton.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (error) {
    setStatus(`The browser didn't allow the microphone: ${error.message}`);
    startButton.disabled = false;
    return;
  }
  ctx = new AudioContext();
  await ctx.audioWorklet.addModule('/static/mic-worklet.js');
  const source = ctx.createMediaStreamSource(stream);
  const capture = new AudioWorkletNode(ctx, 'mic-capture');
  const mute = ctx.createGain();
  mute.gain.value = 0; // keeps the capture node running without playing the mic back
  source.connect(capture);
  capture.connect(mute);
  mute.connect(ctx.destination);
  const block = Math.max(1, Math.floor(ctx.sampleRate / 10)); // ~100 ms per message
  capture.port.onmessage = (event) => onFrame(event.data, block);

  t0 = performance.now();
  ws = new WebSocket(root.dataset.wsUrl);
  ws.addEventListener('message', (event) => {
    try { play(JSON.parse(event.data)); } catch { /* ignore anything that isn't audio */ }
  });
  ws.addEventListener('open', () => {
    setStatus('Listening. Say hello to begin, or click "Begin now".');
    beginButton.disabled = false;
    endButton.disabled = false;
  });
  ws.addEventListener('close', (event) => {
    if (stopped) return;
    if (event.code === 4403) setStatus('This practice session has expired. Reload the page to start a new one.');
    else setStatus('The connection to the interviewer closed. Reload the page to start again.');
    stopAll();
  });
});

beginButton.addEventListener('click', async () => {
  beginButton.disabled = true;
  try { await post(`/api/sessions/${id}/begin`); } catch (error) { setStatus(error.message); beginButton.disabled = false; }
});

endButton.addEventListener('click', async () => {
  endButton.disabled = true;
  beginButton.disabled = true;
  setStatus('Ending…');
  try { await post(`/api/sessions/${id}/end`); } catch (error) { setStatus(error.message); }
  stopAfterPlayback();
});

poll(id, (data) => {
  if (!data) {
    setStatus('This practice session has ended. Reload the page to start a new one.');
    if (!stopped && ws) stopAll();
    return;
  }
  renderTranscript($('transcript'), data.transcript);
  $('speaking').textContent = data.speaking ? `AI: ${data.speaking}` : '';
  if (data.begun) beginButton.disabled = true;
  if (isFinished(data) && ws && !stopped) {
    setStatus('Practice finished. Scoring takes a few minutes; the session page shows the scorecard.');
    stopAfterPlayback();
  }
});
