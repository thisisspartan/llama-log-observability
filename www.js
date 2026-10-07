#!/usr/bin/env node
'use strict';

const http = require('http');
const readline = require('readline');
const fs = require('fs');
const path = require('path');

const argv = process.argv;
const urlIndex = argv.indexOf('--url');
const SERVER_URL = urlIndex !== -1 && argv[urlIndex + 1]
  ? argv[urlIndex + 1].replace(/\/$/, '')
  : (process.env.LLAMA_SERVER_URL || 'http://127.0.0.1:8080').replace(/\/$/, '');
const portIndex = argv.indexOf('--port');
const WEB_PORT = portIndex !== -1 && parseInt(argv[portIndex + 1], 10) > 0
  ? parseInt(argv[portIndex + 1], 10)
  : (parseInt(process.env.WEB_PORT, 10) > 0 ? parseInt(process.env.WEB_PORT, 10) : 3000);
// #15: monitor — local tool; by default listens on localhost ONLY.
// MONITOR_HOST — explicit opt-in for other interfaces (e.g. 0.0.0.0).
const MONITOR_HOST = process.env.MONITOR_HOST || '127.0.0.1';
const REPLAY = process.argv.indexOf('--replay');
const PLAY = process.argv.indexOf('--play');
const SPEED_INDEX = process.argv.indexOf('--speed');
const PLAY_SPEED = SPEED_INDEX !== -1
  ? Math.max(0.1, parseFloat(process.argv[SPEED_INDEX + 1]) || 1)
  : 1;
const RING_MODEL = process.argv.indexOf('--ring-model');
// A5: max pause between lines (sec) in the virtual PLAY timeline.
// 0 = disabled (default): timestamps pass 1:1 / speed.
const MAX_GAP_INDEX = process.argv.indexOf('--max-gap');
const MAX_GAP_MS = MAX_GAP_INDEX !== -1
  ? Math.max(0, parseFloat(process.argv[MAX_GAP_INDEX + 1]) || 0) * 1000
  : 0;
// LIVE: stdin stream of llama-server (without --replay/--play/--ring-model)
const LIVE_MODE = REPLAY === -1 && PLAY === -1 && RING_MODEL === -1;

// Real clock (virtual Date.now in replay/play does not touch the original)
const realNow = Date.now;

// Disk logging with failure protection.
// B5: the stream is created ONLY in LIVE — in --replay/--ring-model lines are not written,
// in --play the log file is not touched (lines are only read from the recorded log).
const LOG_FILE = process.env.LLAMA_LOG_FILE || path.join(process.env.HOME || '.', 'llama-server.log');
// #17: append-only log grows unbounded — size limit + rotation
// (LOG_FILE → LOG_FILE.1, then we write to a new one). LLAMA_LOG_MAX_MB —
// limit in MB (default 50); 0 = rotation disabled.
const LOG_MAX_BYTES = (parseInt(process.env.LLAMA_LOG_MAX_MB, 10) > 0
  ? parseInt(process.env.LLAMA_LOG_MAX_MB, 10) : 50) * 1024 * 1024;
let logSize = 0;
try { if (LIVE_MODE && fs.existsSync(LOG_FILE)) logSize = fs.statSync(LOG_FILE).size; } catch (_) {}
let logStream = LIVE_MODE
  ? fs.createWriteStream(LOG_FILE, { flags: 'a' })
  : { write() {}, on() {} };
logStream.on('error', (err) => {
  console.error('\x1b[31m[LOG STREAM ERROR]\x1b[0m', err.message);
});
function rotateLog() {
  if (!LIVE_MODE || LOG_MAX_BYTES <= 0) return;
  try {
    logStream.end(); // flushes the buffer, then closes
    fs.renameSync(LOG_FILE, LOG_FILE + '.1');
    logSize = 0;
    logStream = fs.createWriteStream(LOG_FILE, { flags: 'a' });
    logStream.on('error', (err) => {
      console.error('\x1b[31m[LOG STREAM ERROR]\x1b[0m', err.message);
    });
  } catch (err) {
    console.error('\x1b[31m[LOG ROTATE ERROR]\x1b[0m', err.message);
  }
}

const MAX_EVENTS = 500;
const MAX_HISTORY = 300;

// Full declarative reset of task state — the single source of the shape
function freshTaskState() {
  return {
    id: null,
    totalPrompt: 0,
    lcpTokens: 0,
    lcpRatio: 0,
    kvRestoredTokens: 0,
    kvRestoredMiB: 0,
    deltaTokens: 0,
    prefillTokens: 0,
    prefillProgress: 0,
    prefillMs: 0,
    prefillTps: 0,
    // H7/#5: prefill is MEASURED (source: the "prompt eval time" line — always
    // a positive measurement, or a POSITIVE n_prompt_tokens_processed from
    // /slots). 0 from /slots is NOT proof of measurement (the slot may not have
    // started prompt-processing yet). Note: a full KV hit gives processed = 1,
    // not 0 — upstream decrements n_past by 1 (TAG_PROMPT_LOGITS:
    // "need to evaluate at least 1 token").
    prefillTimingSeen: false,
    generatedTokens: 0,
    decodeTps: 0,
    decodeTpsPeak: 0,
    decodeMs: 0,
    // SPEC TOTAL (per-task): draft acceptance of ALL speculative implementations
    // (draft-mtp + ngram-mod) — the "draft acceptance" line is printed per-task
    // and reset between tasks. Per-source CUMULATIVE statistics
    // ("statistics draft-mtp:" / "statistics ngram-mod:") live in
    // state.specMtp / state.ngram — a separate session panel.
    specAcceptance: 0,
    specMeanLen: 0,
    // #1: length — from the "acc per pos" line data (upstream resizes the tuple to
    // common_speculative_n_max); we do not invent a fixed 5 positions
    specPosAcc: [],
    specAccepted: null,
    specGenerated: null,
    specInvariantOk: null,
    isTruncated: false,
    // truncated=1 happens in TWO cases (upstream server-context.cpp):
    // (a) overflow: prompt.n_tokens() + 1 >= n_ctx → STOP_TYPE_LIMIT;
    // (b) context shift: "slot context shift, n_keep/n_left/n_discard" —
    // generation continues, but the truncated flag stays sticky (even on a
    // normal EOS after shift). ctxShifted distinguishes (b) from (a).
    ctxShifted: false,
    decodeStarted: false,
    prefillReported: false,
    kvEventEmitted: false,
    cachedFirstSeen: false,
    // PROVENANCE (llama.cpp canon): the source of the cached prefix
    // is NOT derived from "cached n_tokens" — only from event correlation:
    // LCP slot match → 'hot-vram'; found better prompt → 'prompt-cache-ram';
    // neither → 'empty'. restored context checkpoint → the flag below.
    slotSource: null,          // 'hot-vram' | 'prompt-cache-ram' | 'empty'
    checkpointRestore: false,  // context checkpoint restored (Host RAM → VRAM)
    lastProgressStep: 0,
    // Exact token divergence point from [LCP-DEBUG] ... <== DIVERGE
    // (tooltip in REQUEST HISTORY): { pos, tokA, pieceA, tokB, pieceB }
    divergeInfo: null
  };
}

// State shape — the SINGLE source (factories). resetAll() uses the same
// factories, so a "forgotten field" (the eraseLoop incident) is structurally impossible.
// IMPORTANT: the real `player` object (time anchors, timer, lastT/lastDoneT)
// is NOT part of state — state.player is only a projection for UI/SSE.
// playerStart/playerPause/playerRestart/goLive control the real `player`;
// resetAll() resets only the projection (updatePlayerState() re-syncs it).
function freshRamCacheState() {
  return {
    limitMiB: null,
    usedMiB: null,
    promptsCount: null,
    cpCreated: 0,
    cpErased: 0,
    cpCurrent: 0,
    cpMax: null,
    cpUsedMiB: 0,
    stateAt: null,
    promptCacheMiB: null,
    limitTokens: null,
    limitTokensEff: null,
    entries: [],
    entriesAt: null,
    slots: {}
  };
}

// SPECULATIVE — SESSION CUMULATIVE (common_speculative_print_stats:
// counters live for the whole server lifetime, common_speculative_reset is not called).
// #calls(b,g,a) = n_call_begin n_call_draft n_call_accept.
function freshSpecState() {
  return { calls: [0, 0, 0], genDrafts: 0, accDrafts: 0, genTokens: 0, accTokens: 0, meanLen: 0, posAcc: [], invariantOk: null };
}

function freshSessionState() {
  return {
    n_ctx: null, // V1: unconfirmed — will come from /slots (LIVE) or the n_ctx_slot line (play/replay)
    n_ctxSrc: null, // cross-check: which source confirmed n_ctx first
    n_ctxMismatch: null, // cross-check: { sourceA: valueA, sourceB: valueB } — recorded once
    n_prompt_tokens: 0,
    is_processing: false,
    currentStage: 'IDLE',
    _awaitingIdle: false,
    _doneSince: 0,
    _processingStart: 0,
    // /metrics (llama.cpp Prometheus): LIVE only, stays zero in replay/play
    serverMetrics: {
      queueDeferred: 0,          // llamacpp:requests_deferred
      promptTokensTotal: 0,     // llamacpp:prompt_tokens_total
      tokensPredictedTotal: 0,  // llamacpp:tokens_predicted_total
      fetchedAt: 0,
      // #18: /metrics has not answered yet — zeros are NOT data, UI shows N/A
      metricsSeen: false
    },
    ramCache: freshRamCacheState(),
    task: freshTaskState(),
    lastPrefillTps: 0, // M3: last MEASURED prefill t/s — lives between tasks (for SAVED TIME)
    specMtp: freshSpecState(),
    ngram: freshSpecState(),
    // SPEC CONFIG — the real launch configuration from the startup log
    // ("adding speculative implementation 'draft-mtp'" → "- n_max=N").
    // null = the server did not print the startup lines (log from mid-session).
    specConfig: { draftNMax: null },
    history: [],
    events: [],
    // C1/B2: array versions — the snapshot sends events/history only on change
    eventsRev: 0,
    historyRev: 0,
    // A1: player state (PLAY / load-log); stays zero in LIVE.
    // PROJECTION ONLY — the source of truth: the `player` object (see above).
    player: { paused: false, speed: PLAY_SPEED, idx: 0, total: 0, finished: false },
    // CHECKPOINT LOOP watchdog: the current chain of identical erase lines
    eraseLoop: null
  };
}

const state = Object.assign(freshSessionState(), {
  serverUrl: SERVER_URL,
  serverOnline: false,
  // Server mode: 'LIVE' | 'PLAYBACK' | 'REPLAY' (the client dims the player buttons in LIVE)
  mode: 'LIVE',
  // "⏭ TO LIVE" is possible only with a live stdin stream (launch without --play/--replay):
  // from PLAYBACK after load-log in a --play process LIVE would be dead
  liveCapable: LIVE_MODE
});

// LCP-match buffers, KEYED BY slot_id (not global): each slot
// picks its own candidate independently. Upstream (server-context.cpp) does
// the LCP choice per server_task/slot; a global queue with --parallel>1 glued
// task B's LCP to task A under interleaving. The "checking sim"/"selected slot"/
// "new prompt" lines carry "id <slot>" — that is the correlation key.
const pendingLcp = new Map();    // slotId -> {ratio, tokens, pct, slotId, viaRamCache}
// Best candidate from the RAM cache (srv load, stage 2) before "found better prompt".
// srv load lines carry NO slot/task id — attributed to lastSelectionSlot.
const pendingRamLcp = new Map(); // slotId -> {fKeep, fSim, ratio, tokens, pct, viaRamCache}
// Baseline of the RAM-cache choice for THIS slot (upstream server-task.cpp):
// "looking for better prompt, base f_keep = X, f_sim = Y" — f_keep/f_sim
// of the slot's CURRENT prompt (empty slot: f_keep = -1.0 → any candidate
// with f_keep >= 0.25 wins). A candidate is accepted only if strictly
// better on BOTH metrics against the baseline; on a win the baseline
// is updated with the candidate (upstream sequential pass).
const pendingRamBase = new Map(); // slotId -> {fKeep, fSim}
// Slot chosen by LRU (no similar prefix) — Cold Start
const pendingLru = new Map();    // slotId -> {slotId}
// Slot of the last selection line ("checking sim"/"selected slot") — for attributing
// srv load lines that carry no slot id of their own.
let lastSelectionSlot = null;
// Prompt cache record buffer: opened by the "cache state:" line with
// the expected record count (promptsCount), committed by the counter (N2)
let cacheCollecting = null, cacheExpected = 0, cacheStartedAt = 0;
// A3: time of the last log line (epoch ms from the "[ISO] " prefix) — the single
// source of ts for events and history time. In LIVE there is no prefix → null → Date.now().
let currentLogTs = null;
// #7: buffer of service prompt_save events. The prompt_save line is printed
// by the server at task boundaries (prompt save BEFORE the new one) — gluing it
// to the PREVIOUS task's card would be wrong. We buffer it and on the next
// "new prompt" flush it into the NEW task (chronologically they are older than INPUT).
const pendingSysEvents = [];
// SPEC CONFIG: the line "adding speculative implementation 'draft-mtp'"
// means the NEXT line carries "- n_max=N, ..." — we wait for it.
let pendingSpecImpl = null;

const sseClients = new Set();
let stateDirty = false;
// C1: versions of the last INCLUDED arrays are stored PER CLIENT
// (res._evRev / res._histRev) — one client does not shift the other snapshots.
// full=true (connect to /events, /api/state) — always full, does not touch versions.
function snapshotJson(full = false, evRev = -1, histRev = -1) {
  state.serverNow = Date.now();
  const out = Object.assign({}, state);
  if (!full && state.eventsRev === evRev) out.events = [];
  if (!full && state.historyRev === histRev) out.history = [];
  return JSON.stringify(out);
}
function broadcastState() {
  stateDirty = true;
}
function flushState() {
  if (!stateDirty) return;
  stateDirty = false;
  // Without clients we do not build the snapshot: a new client gets the full
  // snapshot when connecting to /events
  if (sseClients.size === 0) return;
  for (const client of sseClients) {
    const payload = `data: ${snapshotJson(false, client._evRev, client._histRev)}\n\n`;
    client._evRev = state.eventsRev;
    client._histRev = state.historyRev;
    try {
      client.write(payload);
      if (typeof client.flush === 'function') client.flush();
    } catch (_) {
      sseClients.delete(client);
    }
  }
}
setInterval(flushState, 100);
function markOffline() {
  if (state.serverOnline) {
    state.serverOnline = false;
    broadcastState();
  }
}

// Task binding: true if the line names a task id different from the current one
// (protection against desync under a fast request stream / foreign slots)
function isForeignTask(line) {
  const m = line.match(/task\s+(\d+)/);
  return m !== null && state.task.id !== null && m[1] !== state.task.id;
}

function addEvent(stage, tag, titleKey, detailKey, params, explicitTaskId = null, lifecycle = true) {
  // A3: time of the log line (play/replay), in LIVE — the real clock
  const ts = currentLogTs != null ? new Date(currentLogTs).toLocaleTimeString() : new Date().toLocaleTimeString();
  if (lifecycle) state.currentStage = stage;
  const tid = explicitTaskId !== null ? String(explicitTaskId) : (state.task.id ? String(state.task.id) : 'SYS');
  const evId = Date.now() + Math.random();
  // i18n: the server stores DICTIONARY KEYS + params; the client renders via t()
  state.events.unshift({ ts, stage, tag, titleKey, detailKey, params: params || {}, id: evId, taskId: tid });
  if (state.events.length > MAX_EVENTS) state.events.pop();
  state.eventsRev++; // B2: any feed change — a new version
  broadcastState();
  return evId;
}

// V1 cross-check: n_ctx from different sources (log lines vs /slots) must
// match. The first source wins; a mismatch is recorded ONCE
// (flag + event + console.error) — not on every poll.
function confirmNctx(value, src) {
  if (!Number.isFinite(value) || value <= 0) return;
  if (state.n_ctx == null) {
    state.n_ctx = value;
    state.n_ctxSrc = src;
    return;
  }
  if (value !== state.n_ctx && state.n_ctxMismatch == null) {
    state.n_ctxMismatch = { [state.n_ctxSrc]: state.n_ctx, [src]: value };
    addEvent('SYS', 'NCTX_MISMATCH', 'ev_nctx_title', 'ev_nctx_detail',
      { a: state.n_ctxSrc + '=' + state.n_ctx, b: src + '=' + value }, null, false);
    console.error(`[NCTX_MISMATCH] ${state.n_ctxSrc}=${state.n_ctx} vs ${src}=${value}`);
  }
}

// Supplements the SLOT SELECT event with the KV source once it becomes known
// (RAM checkpoint / hot VRAM slot / cold start) — the corresponding lines
// arrive AFTER the slot selection
function annotateSlotSelect(taskId, sourceKey) {
  if (taskId == null) return;
  const ev = state.events.find(e => e.stage === 'PREFIX' && e.tag === 'SLOT SELECT' && e.taskId === String(taskId));
  // E2: idempotency — if the source is already recorded, do not write again
  // (otherwise the source would be overwritten on restored→cached)
  if (ev && !ev.params.sourceKey) {
    ev.params.sourceKey = sourceKey;
    state.eventsRev++; // B2: params change of an existing event
  }
}

// Single history record: base shape from state.task + branch specifics (extra)
function pushHistory(extra) {
  const lcp = state.task.lcpTokens || 0;
  const kvReuse = state.task.kvRestoredTokens || 0;
  // CP SKEW: the exact LCP went further than the actually restored prefix —
  // the difference was recomputed in prefill (overpay for the checkpoint step).
  // On cold start (kvReuse=0 / slotSource='empty') SKEW does not exist —
  // the whole prompt was recomputed, not the LCP "tail": we do not show a phantom
  const cpSkew = (kvReuse > 0 && state.task.slotSource !== 'empty') ? Math.max(0, lcp - kvReuse) : 0;
  state.history.unshift(Object.assign({
    id: String(state.task.id || '—'),
    input: state.task.totalPrompt,
    lcp: lcp,
    kvReuse: kvReuse, // cachedTokens N (length of the cached prefix)
    cpSkew: cpSkew,
    divergeInfo: state.task.divergeInfo || null,
    slotSource: state.task.slotSource || null,
    checkpointRestore: Boolean(state.task.checkpointRestore),
    reprocessReason: state.task.reprocessReason || null,
    // H7: prefillTimingSeen — 0 can be a measured value (e.g. after
    // a context shift), so the status comes from the flag, not from the value's truthiness
    prefill: state.task.prefillTimingSeen ? state.task.prefillTokens : (state.task.prefillTokens || state.task.deltaTokens),
    prefillSource: (state.task.prefillTimingSeen || state.task.prefillTokens) ? 'measured' : 'estimated',
    prefillSpeed: state.task.prefillTps || state.lastPrefillTps || null,
    decode: state.task.generatedTokens,
    speed: state.task.decodeTps,
    total: state.task.totalPrompt + state.task.generatedTokens,
    isTruncated: Boolean(state.task.isTruncated),
    ctxShifted: Boolean(state.task.ctxShifted),
    specAccepted: state.task.specAccepted || null,
    specGenerated: state.task.specGenerated || null,
    specAcceptance: state.task.specAcceptance || null,
    specMeanLen: state.task.specMeanLen || null,
    specPosAcc: state.task.specPosAcc.slice(),
    specInvariantOk: state.task.specInvariantOk,
    time: currentLogTs != null ? new Date(currentLogTs).toLocaleTimeString() : new Date().toLocaleTimeString()
  }, extra));
  if (state.history.length > MAX_HISTORY) state.history.pop();
  state.historyRev++; // B2: a new history record
}

function initNewTask(id, promptTokens, slotId) {
  // Reset the UNCOMMITTED RAM candidate of THIS slot (not globally):
  // a foreign slot must not lose its own candidate.
  if (slotId != null) { pendingRamLcp.delete(slotId); pendingRamBase.delete(slotId); }
  lastStallWarnSec = 0;
  state._awaitingIdle = false;
  // A3: task start — time of the log line (play/replay), in LIVE — the real clock
  state._processingStart = currentLogTs != null ? currentLogTs : Date.now();
  state.task = freshTaskState();
  state.task.id = String(id);
  state.task.totalPrompt = promptTokens;

  // Publish LCP exactly once, strictly inside the new task's card.
  // Take the candidate of THIS slot (slot_id from the new prompt line) — not the global one.
  // Fallback: an unattributed candidate (srv load without a preceding "checking sim")
  // sits under the null key — the next new prompt picks it up.
  let lcp = slotId != null ? pendingLcp.get(slotId) : null;
  let lcpKey = slotId;
  if (lcp == null) { lcp = pendingLcp.get(null); lcpKey = null; }
  if (lcp) {
    pendingLcp.delete(lcpKey);
    state.task.lcpTokens = lcp.tokens || Math.round(promptTokens * lcp.ratio);
    // Exact ratio against the NEW prompt (upstream f_sim = lcp / new_prompt_size).
    // lcp.ratio from the log line — rounded %.3f (and for the RAM candidate this is
    // f_sim against the new prompt, not against the candidate length) — the recompute gives
    // the exact value when lcpTokens is known.
    state.task.lcpRatio = promptTokens > 0 ? state.task.lcpTokens / promptTokens : lcp.ratio;
    const pct = (state.task.lcpRatio * 100).toFixed(1);
    // Provenance: LCP match with an existing slot = hot-vram;
    // candidate from the prompt cache (srv load, "found better prompt") = prompt-cache-ram
    state.task.slotSource = lcp.viaRamCache ? 'prompt-cache-ram' : 'hot-vram';
    addEvent('PREFIX', 'PREFIX', 'ev_lcp_title',
      state.task.lcpTokens > 0 ? 'ev_lcp_detail_tok' : 'ev_lcp_detail',
      { pct, tok: state.task.lcpTokens.toLocaleString() }, String(id));
    // SLOT SELECT: which slot was chosen by LCP and how many tokens matched.
    // The source (RAM checkpoint / VRAM hot slot / cold start) is appended later —
    // the corresponding lines arrive after new prompt
    addEvent('PREFIX', 'SLOT SELECT', 'ev_slot_lcp_title',
      lcp.slotId != null ? 'ev_slot_lcp_detail_slot' : 'ev_slot_lcp_detail',
      { slot: lcp.slotId, lcp: state.task.lcpTokens.toLocaleString(), prompt: promptTokens.toLocaleString(), pct }, String(id));
  } else {
    state.task.lcpTokens = 0;
    state.task.lcpRatio = 0;
    const lru = slotId != null ? pendingLru.get(slotId) : null;
    if (lru) {
      pendingLru.delete(slotId);
      state.task.slotSource = 'empty';
      addEvent('PREFIX', 'SLOT SELECT', 'ev_slot_lru_title', 'ev_slot_lru_detail', { slot: Number(lru.slotId) }, String(id));
    }
  }

  state.task.deltaTokens = Math.max(0, promptTokens - state.task.lcpTokens);
}

// In --play stdin is not used — lines are fed from the file by timestamps
const rl = PLAY === -1
  ? readline.createInterface({ input: process.stdin, terminal: false })
  : null;

// Sweeps slots evicted/resynced (removedAt) more than 2 s ago (log time)
function sweepGone() {
  let changed = false;
  for (const k in state.ramCache.slots) {
    const sl = state.ramCache.slots[k];
    const gone = sl.evictedAt || sl.removedAt;
    if (gone && Date.now() - gone > 2000) { delete state.ramCache.slots[k]; changed = true; }
  }
  if (changed) recomputeCp(); // H10: physical removal must recompute cpCurrent/cpUsedMiB
}

// Reconciliation N of M (V6: PER-SLOT). Upstream (server-context.cpp create_checkpoint):
// checkpoints live in slot.prompt.checkpoints — a list PER SLOT; N in
// "created context checkpoint N of M" — the position in THIS slot's list,
// M (n_ctx_checkpoints) — the limit PER SLOT (per-slot cap). So reconciliation and eviction
// of the extra one happen only inside the slot that created the checkpoint; foreign slots are not touched.
// N==1 (prompt change) or a recompute — the extra oldest of THIS slot get
// removedAt (reason resync). Shortfall (the monitor attached mid-session) —
// N is authoritative for this slot, we do not invent the missing ones.
// cpCurrent/cpUsedMiB — the sum over ALL slots (the shelf is shared, ownership is per-slot).
// B3: a single recompute. resync: N is authoritative for its slot (otherLive + max(len, N));
// erase: a plain live count. slotId may be null (a log without "id N |") —
// then "own slot" = records without an id, comparison s.slot === slotId without a null check.
function recomputeCp(opts = {}) {
  const { resync = false, slotId = null, N = 0 } = opts;
  const live = Object.values(state.ramCache.slots).filter(s => !s.evictedAt && !s.removedAt);
  let count;
  if (resync && N > 0) {
    const thisSlot = live.filter(s => s.slot === slotId);
    count = (live.length - thisSlot.length) + Math.max(thisSlot.length, N);
  } else {
    count = live.length;
  }
  state.ramCache.cpCurrent = count;
  state.ramCache.cpUsedMiB = live.reduce((a, s) => a + s.size, 0);
}

function resyncCheckpoints(slotId, N, keepKey) {
  const slots = state.ramCache.slots;
  const now = Date.now();
  const live = Object.values(slots).filter(s => !s.evictedAt && !s.removedAt);
  const thisSlot = live.filter(s => s.slot === slotId);
  if (thisSlot.length > N) {
    thisSlot.sort((a, b) => a.createdAt - b.createdAt);
    // B10a: a just-created slot (keepKey) is not removed — with equal
    // createdAt the sort order is arbitrary
    const removable = thisSlot.filter(s => s.key !== keepKey);
    const excess = thisSlot.length - N;
    for (let i = 0; i < excess && i < removable.length; i++) {
      removable[i].removedAt = now;
      removable[i].removeReason = 'resync';
    }
  }
  recomputeCp({ resync: true, slotId, N }); // B3: N is authoritative for its slot
}

// Prompt cache record: the address may lack 0x and be uppercase (B7)
const ENTRY_RE = /update:\s+-\s+prompt\s+((?:0x)?[0-9a-fA-F]+):\s+(\d+)\s+tokens,\s+checkpoints:\s+(\d+),\s+([\d.]+)\s+MiB/;

// Commit of the collected prompt cache records: comparison with the old list by key
// (address). The first commit — baseline: createdAt=null (B6: no "growing" flash
// at monitor start). New key → createdAt=now; key exists → update
// tokens/sizeMiB, touchedAt only on a real change (B1); an old key
// is absent → removedAt=now (lives for another 2 s).
function commitCacheEntries(list) {
  const now = Date.now();
  const baseline = state.ramCache.entriesAt === null; // first commit — the reference point
  const old = state.ramCache.entries;
  const byKey = new Map(old.filter(e => !e.removedAt).map(e => [e.key, e]));
  const seen = new Set();
  for (const e of list) {
    seen.add(e.key);
    const prev = byKey.get(e.key);
    if (!prev) {
      old.push({ key: e.key, tokens: e.tokens, checkpoints: e.checkpoints, sizeMiB: e.sizeMiB,
                 createdAt: baseline ? null : now, touchedAt: null, removedAt: null, restoreCount: 0 });
      continue;
    }
    const changed = prev.tokens !== e.tokens || prev.checkpoints !== e.checkpoints ||
                    Math.abs(prev.sizeMiB - e.sizeMiB) > 0.001;
    prev.tokens = e.tokens; prev.checkpoints = e.checkpoints; prev.sizeMiB = e.sizeMiB;
    if (changed) prev.touchedAt = now; // only on a real change
  }
  for (const e of old) if (!e.removedAt && !seen.has(e.key)) e.removedAt = now;
  state.ramCache.entries = old.filter(e => !e.removedAt || now - e.removedAt <= 2000);
  state.ramCache.entriesAt = now;
  broadcastState();
}

// SPEC invariant (upstream, one statistical series):
//   mean_acc_len = 1 + Σ acc_rate_per_pos
// meanLen is printed with 2 decimals (±0.005), each rate — with 3 decimals (±0.0005)
// → tolerance 0.005 + 0.0005·N (+1e-9 against FP accumulation). null = cannot
// check (no data), true/false = the result.
function specInvariantOk(meanLen, rates) {
  if (!Number.isFinite(meanLen) || !Array.isArray(rates) || rates.length === 0) return null;
  const sum = rates.reduce((a, x) => a + (parseFloat(x) || 0), 0);
  const tol = 0.005 + rates.length * 0.0005 + 1e-9;
  return Math.abs(meanLen - (1 + sum)) <= tol;
}

function handleLine(line) {
  if (!line) return;
  if (line.includes('update_slots: all slots are idle')) return;

  // Prompt cache records: the "cache state:" line opens the buffer with the expected
  // record count (promptsCount). Commit by the counter: on the N-th record or on
  // "prompt cache update took" (only if the list is full). N2: a stray
  // line between records does NOT commit the buffer — no false removedAt.
  if (cacheCollecting !== null) {
    const mEntry = line.match(ENTRY_RE);
    if (mEntry) {
      cacheCollecting.push({ key: mEntry[1], tokens: parseInt(mEntry[2], 10), checkpoints: parseInt(mEntry[3], 10), sizeMiB: parseFloat(mEntry[4]) });
      if (cacheCollecting.length >= cacheExpected) { commitCacheEntries(cacheCollecting); cacheCollecting = null; }
      return;
    }
    if (line.includes('prompt cache update took')) {
      if (cacheCollecting.length === cacheExpected) commitCacheEntries(cacheCollecting); // an incomplete list is not committed
      cacheCollecting = null;
      return;
    }
    if (Date.now() - cacheStartedAt > 3000) cacheCollecting = null; // reset by log time
    // other lines — normal processing, the buffer is not touched
  }

  // [LCP-DEBUG] — service lines of the server (after "new prompt", before the KV lines):
  //   summary  "slot_tokens=N input_tokens=M lcp=L" — the EXACT LCP (more
  //            authoritative than the estimate from "checking sim") → we override lcpTokens/deltaTokens;
  //   DIVERGE "i=P A=tokA pieceA B=tokB pieceB <== DIVERGE" — the exact
  //            divergence point (REQUEST HISTORY tooltip).
  // They do not go into the event feed — only diagnostics of the current task.
  if (line.includes('[LCP-DEBUG]')) {
    const mDiv = line.match(/\[LCP-DEBUG\]\s+i\s*=\s*(\d+)\s+A\s*=\s*(\d+)\s+(.*?)\s+B\s*=\s*(\d+)\s+(.*?)\s+<== DIVERGE/);
    if (mDiv) {
      const tidM = line.match(/task\s+(\d+)/);
      if (tidM && state.task.id != null && tidM[1] === String(state.task.id)) {
        state.task.divergeInfo = {
          pos: parseInt(mDiv[1], 10),
          tokA: mDiv[2], pieceA: mDiv[3].trim(),
          tokB: mDiv[4], pieceB: mDiv[5].trim()
        };
        broadcastState();
      }
    } else {
      const mSum = line.match(/\[LCP-DEBUG\]\s+slot_tokens\s*=\s*(\d+)\s+input_tokens\s*=\s*(\d+)\s+lcp\s*=\s*(\d+)/);
      if (mSum) {
        const tidM = line.match(/task\s+(\d+)/);
        if (tidM && state.task.id != null && tidM[1] === String(state.task.id)) {
          state.task.lcpTokens = parseInt(mSum[3], 10);
          state.task.lcpRatio = state.task.totalPrompt > 0 ? state.task.lcpTokens / state.task.totalPrompt : 0;
          state.task.deltaTokens = Math.max(0, state.task.totalPrompt - state.task.lcpTokens);
          broadcastState();
        }
      }
    }
    return;
  }

  // SPEC CONFIG — startup lines (common_speculative_init):
  // "adding speculative implementation 'draft-mtp'" → the NEXT line
  // carries "- n_max=N, ...". The real --spec-draft-n-max value comes
  // from the log — no hardcoded position counts.
  if (line.includes("adding speculative implementation 'draft-mtp'")) {
    pendingSpecImpl = 'draft-mtp';
    return;
  }
  if (pendingSpecImpl === 'draft-mtp') {
    const mN = line.match(/n_max\s*=\s*(\d+)/);
    if (mN) state.specConfig.draftNMax = parseInt(mN[1], 10);
    pendingSpecImpl = null;
    return;
  }

  // V1: the server startup line — n_ctx_slot, printed by the server itself
  // ("srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'")
  if (line.includes('initializing,') && line.includes('n_ctx_slot =')) {
    const m = line.match(/n_ctx_slot\s*=\s*(\d+)/);
    if (m) {
      // A new server life: the old n_ctx is stale — reset the cross-check
      // (a mismatch is valid only WITHIN one life: init/prompt//slots)
      state.n_ctx = null;
      state.n_ctxSrc = null;
      state.n_ctxMismatch = null;
      confirmNctx(parseInt(m[1], 10), 'init');
    }
    return;
  }

  // M5: fast path — a decode line (the majority of log lines). The double marker
  // 'tg =' + 'n_gen ='/'n_decoded =' is specific to decode; cacheCollecting
  // and initializing are handled above. A pure reorder of branch 9 — behavior does
  // not change (golden diff empty).
  if (line.includes('tg =') && (line.includes('n_gen =') || line.includes('n_decoded ='))) {
    if (isForeignTask(line)) return; // a foreign task (desync) — do not touch the current state
    const m = line.match(/(?:n_gen|n_decoded)\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+)\s*t\/s(?:,\s*tg_3s\s*=\s*([\d.]+)\s*t\/s)?/);
    if (m) {
      state.currentStage = 'DECODE';
      // Monotonicity: a log line may lag behind /slots (n_decoded,
      // 500 ms polling) — a backward jump jerked the decode arc edge. The final
      // value is set by the timing line (DONE) without this guard.
      const g = parseInt(m[1], 10);
      if (g > state.task.generatedTokens) state.task.generatedTokens = g;
      state.task.decodeTps = parseFloat(m[2]);
      if (m[3]) state.task.decodeTpsPeak = parseFloat(m[3]);

      if (!state.task.decodeStarted) {
        state.task.decodeStarted = true;
        // Fallback: if the "prompt eval time" line did not make it into the log,
        // close the prefill with an estimate (totalPrompt - LCP)
        if (!state.task.prefillReported) {
          state.task.prefillReported = true;
          const dTokens = state.task.deltaTokens || state.task.totalPrompt;
          addEvent('PREFILL', 'PREFILL', 'ev_prefill_done_title', 'ev_prefill_done_est', { tok: Number(dTokens).toLocaleString() });
        }
        addEvent('DECODE', 'DECODE', 'ev_decode_start_title', 'ev_decode_start_detail', { tps: m[2] });
      } else {
        broadcastState();
      }
    }
    return; // M5b: the decode line is fully handled — do not walk the rest of the chain
  }

  // 1. Host RAM hit (srv load) — the search in the RAM cache.
  // srv load lines carry NO slot/task id — we attribute them to lastSelectionSlot
  // (the slot of the last selection line, see below).
  // 1a. Baseline (upstream server-task.cpp): f_keep/f_sim of the CURRENT
  // slot prompt BEFORE the cache walk. The line is printed BEFORE the candidates.
  // "I srv load:  - looking for better prompt, base f_keep = 0.800, f_sim = 0.800"
  if (line.includes('looking for better prompt')) {
    const m = line.match(/base f_keep\s*=\s*(-?[\d.]+),\s*f_sim\s*=\s*(-?[\d.]+)/);
    if (m && lastSelectionSlot != null) {
      pendingRamBase.set(lastSelectionSlot, { fKeep: parseFloat(m[1]), fSim: parseFloat(m[2]) });
    }
  }
  // Line: "I srv          load:  - found better prompt with f_keep = 1.000, f_sim = 0.999"
  else if (line.includes('found better prompt')) {
    const slot = lastSelectionSlot;
    // Upstream (server-task.cpp load): prompt = std::move(it_best->prompt) —
    // the checkpoint list moves TOGETHER with the prompt, there is no separate line.
    // The old shelf labels of THIS slot (checkpoints of the previous prompt) are stale
    // — we remove them (removedAt, reason prompt-replace). The candidate's inherited checkpoints
    // were not on the shelf (they lived as a RAM-prompt, not as a slot-cp);
    // the counter reconciles on the next created line's N of M (N is authoritative).
    if (slot != null) {
      const now = Date.now();
      for (const k in state.ramCache.slots) {
        const sl = state.ramCache.slots[k];
        if (sl.evictedAt || sl.removedAt) continue;
        if (sl.slot !== Number(slot)) continue;
        sl.removedAt = now;
        sl.removeReason = 'prompt-replace';
      }
      recomputeCp();
    }
    // Commit the best RAM candidate of THIS slot — it overrides
    // the stage-1 LCP (checking sim) for the same slot.
    if (slot != null && pendingRamLcp.has(slot)) {
      pendingLcp.set(slot, pendingRamLcp.get(slot));
      pendingRamLcp.delete(slot);
    } else {
      // Fallback: no exact candidate — take f_sim from the line itself
      const m = line.match(/(?:f_)?sim\s*=\s*([\d.]+)/);
      if (m) {
        const r = parseFloat(m[1]);
        pendingLcp.set(slot, { ratio: r, tokens: 0, pct: (r * 100).toFixed(1), slotId: slot, viaRamCache: true });
      }
    }
  }
  // 1b. srv load candidates: "prompt with length   47328, lcp =   47328, f_keep = 1.000, f_sim = 0.999"
  else if (line.includes('prompt with length') && line.includes('lcp =')) {
    const m = line.match(/prompt with length\s+(\d+),\s*lcp\s*=\s*(\d+),\s*f_keep\s*=\s*([\d.]+),\s*f_sim\s*=\s*([\d.]+)/);
    if (m) {
      const tokens = parseInt(m[2], 10);
      const fKeep = parseFloat(m[3]);
      const fSim = parseFloat(m[4]);
      // ratio = f_sim (lcp / new_prompt_size, upstream) — NOT lcp/candidate length:
      // that is f_keep (lcp / candidate length), a different quantity. The exact
      // lcpRatio against the new prompt is recomputed in initNewTask
      // (lcpTokens / promptTokens).
      const ratio = fSim;
      const slot = lastSelectionSlot;
      // Upstream (server-task.cpp): a candidate with f_keep < 0.25 is discarded
      // ("don't trash large prompts"); the one that STRICTLY improves
      // BOTH f_keep and f_sim against the BASELINE is chosen — f_keep/f_sim of the CURRENT
      // slot prompt (the "looking for better prompt, base ..." line), not
      // against the previous candidate. On a win the baseline is updated
      // with the candidate (sequential pass). Without the base line (old builds)
      // the line = -1/-1: the first admissible candidate wins.
      if (fKeep >= 0.25) {
        const base = (slot != null && pendingRamBase.get(slot)) || { fKeep: -1, fSim: -1 };
        if (base.fKeep < fKeep && base.fSim < fSim) {
          // viaRamCache: this LCP came from the prompt cache (Host RAM), not from
          // a hot VRAM slot — task provenance, not a "hot slot"
          pendingRamLcp.set(slot, { fKeep, fSim, ratio, tokens, pct: (ratio * 100).toFixed(1), viaRamCache: true });
          if (slot != null) pendingRamBase.set(slot, { fKeep, fSim });
        }
      }
    }
  }

  // 1c. LRU: no similar prefix — the slot was chosen by LRU (Cold Start)
  if (line.includes('selected slot by LRU')) {
    const m = line.match(/\bid\s+(\d+)/);
    const slot = m ? m[1] : '0';
    lastSelectionSlot = slot;
    pendingLcp.delete(slot);
    pendingLru.set(slot, { slotId: slot });
  }

  // 2. PREFIX / LCP (buffering until the new task's ID appears)
  if (line.includes('checking sim =') || line.includes('f_sim_best =')) {
    const m = line.match(/(?:checking sim|sim_best|f_sim_best)\s*=\s*([\d.]+)(?:\s*\((\d+)\/(\d+)\))?/);
    if (m) {
      const ratio = parseFloat(m[1]);
      const tokens = m[2] ? parseInt(m[2], 10) : 0;
      const pct = (ratio * 100).toFixed(1);
      const slotM = line.match(/\bid\s+(\d+)/);
      const slot = slotM ? slotM[1] : lastSelectionSlot;
      if (slot != null) lastSelectionSlot = slot;

      // Keep the best result for the next task of THIS slot, without
      // creating events in the old one. The key — slot_id (not global).
      if (line.includes('sim_best')) {
        // f_sim_best without (N/M): the exact count — from "checking sim" of the same
        // candidate. Both lines print ONE AND THE SAME float f_sim_best
        // (server-context.cpp: SLT_TRC "checking sim" → SLT_INF "sim_best"),
        // so equality of the ratio (after %.3f) — an exact candidate match.
        // Without a match (the checking sim line did not arrive — trace level)
        // tokens = 0 → the estimate in initNewTask.
        const prev = pendingLcp.get(slot);
        const exactTokens = (prev && prev.ratio === ratio) ? prev.tokens : 0;
        pendingLcp.set(slot, { ratio, tokens: exactTokens, pct, slotId: slot });
      } else {
        const cur = pendingLcp.get(slot);
        if (!cur || ratio >= cur.ratio) {
          pendingLcp.set(slot, { ratio, tokens, pct, slotId: slot });
        }
      }
    }
  }
  // 3. INPUT (task start)
  else if (line.includes('new prompt,') && (line.includes('task.n_tokens') || line.includes('n_prompt_tokens'))) {
    const m = line.match(/(?:task\s+(\d+).*?)?new prompt,.*?(?:task\.n_tokens|n_prompt_tokens)\s*=\s*(\d+)/);
    if (m) {
      const tid = m[1] || '0';
      const promptTokens = parseInt(m[2], 10);
      // slot_id from the line ("id 0 |") — the correlation key for this slot's LCP candidate
      const mSlot = line.match(/\bid\s+(\d+)\s*\|/);
      const slotId = mSlot ? mSlot[1] : null;
      // #7: service prompt_save events of the previous task are flushed into the NEW
      // task BEFORE INPUT — chronologically (the feed is bottom-up) they are older than INPUT
      for (const pe of pendingSysEvents) addEvent(pe.stage, pe.tag, pe.titleKey, pe.detailKey, pe.params, tid, false);
      pendingSysEvents.length = 0;
      // E1: INPUT is registered FIRST (before initNewTask) — with unshift in
      // state.events the feed chronology is bottom-up: [INPUT] -> [PREFIX] ->
      // [SLOT SELECT] -> [KV REUSE]
      addEvent('INPUT', 'INPUT', 'ev_input_title', 'ev_input_detail', { tid, tok: promptTokens.toLocaleString() }, tid);
      initNewTask(tid, promptTokens, slotId);
      state.is_processing = true;
      // Defect 5: INPUT is fixed immediately — no IDLE flicker between lines
      state.currentStage = 'INPUT';
      // B9/V1: n_ctx from the log line — the value printed by the server itself
      // (confirmed), valid in all modes; in LIVE /slots confirms it
      const mCtx = line.match(/n_ctx_slot\s*=\s*(\d+)/);
      if (mCtx) confirmNctx(parseInt(mCtx[1], 10), 'prompt');
      console.log(`\x1b[36m⚡ [INPUT]\x1b[0m Request #${tid}: received ${promptTokens} tokens`);
    }
  } 
  // 4. KV REUSE (context checkpoints are physically lifted from Host RAM into VRAM!)
  else if (line.includes('restored context checkpoint')) {
    if (isForeignTask(line)) return;
    // n_past — tokens actually applied in the slot (the true "restored");
    // n_tokens — the checkpoint snapshot size (the key in the RAM cache). C++: n_past ≤ n_tokens.
    // On a partial LCP (a break inside the checkpoint) n_past < n_tokens — we take n_past.
    const mPast = line.match(/n_past\s*=\s*(\d+)/);
    const mSnap = line.match(/n_tokens\s*=\s*(\d+)/);
    const mSize = line.match(/size\s*=\s*([\d.]+)\s*MiB/);
    const snapshotTokens = mSnap ? parseInt(mSnap[1], 10) : 0;
    const tokens = mPast ? parseInt(mPast[1], 10) : snapshotTokens; // n_past → fallback n_tokens
    const size = mSize ? mSize[1] : '0';

    state.task.kvRestoredTokens = tokens;   // cachedTokens N: length of the restored prefix
    state.task.kvRestoredMiB = parseFloat(size);
    state.task.checkpointRestore = true;    // provenance: context checkpoint (Host RAM → VRAM)
    state.task.deltaTokens = Math.max(0, state.task.totalPrompt - state.task.kvRestoredTokens);

    // Context Checkpoints (--ctx-checkpoints) are stored in Host RAM, BUT in
    // slot.prompt.checkpoints — a separate store, NOT limited
    // --cache-ram (that is the server_prompt_cache budget — a different mechanism).
    const tag = 'KV REUSE RAM';

    // If a KV_RESTORE event for this task managed to be set before this
    // (a premature CACHED PREFIX from the first cached line), we replace it with
    // the real lift from RAM — splice by stage, not by the event text
    if (state.task.kvEventEmitted) {
      const inPlaceIdx = state.events.findIndex(
        e => e.stage === 'KV_RESTORE' && e.taskId === String(state.task.id)
      );
      if (inPlaceIdx !== -1) { state.events.splice(inPlaceIdx, 1); state.eventsRev++; } // B2
    }
    state.task.kvEventEmitted = true;
    annotateSlotSelect(state.task.id, 'src_checkpoint');
    addEvent('KV_RESTORE', tag, 'ev_kv_restore_title', 'ev_kv_restore_detail', { tok: tokens.toLocaleString(), size });
    // Pulse on the checkpoint shelf: this slot was just lifted from RAM.
    // H8: exact identity — pos_min/pos_max from the restored line (upstream
    // prints them); the checkpoint is stored under its full size, even if
    // applied partially (n_past < n_tokens). Fallback (old builds without
    // pos) — the snapshot n_tokens, then fuzzy ±500.
    // V6: checkpoints ownership is per-slot (slot.prompt.checkpoints) — the exact
    // search and fuzzy only inside the slot of the line (id N |).
    {
      const mSlot = line.match(/\bid\s+(\d+)\s*\|/);
      const slotId = mSlot ? parseInt(mSlot[1], 10) : null;
      const mPosMin = line.match(/pos_min\s*=\s*(\d+)/);
      const mPosMax = line.match(/pos_max\s*=\s*(\d+)/);
      let sl = null;
      if (mPosMin && mPosMax) {
        const k = slotId != null ? slotId + ':' + mPosMin[1] + ':' + mPosMax[1]
                                 : mPosMin[1] + ':' + mPosMax[1];
        sl = state.ramCache.slots[k];
        if (sl && (sl.evictedAt || sl.removedAt)) sl = null; // a dead one does not pulse
      }
      if (!sl) {
        const keyOf = t => slotId != null ? slotId + ':' + t : String(t);
        sl = state.ramCache.slots[keyOf(snapshotTokens)] || state.ramCache.slots[keyOf(snapshotTokens + 1)];
        if (sl && (sl.evictedAt || sl.removedAt)) sl = null;
      }
      if (sl) { sl.lastRestored = Date.now(); sl.restoreCount = (sl.restoreCount || 0) + 1; }
      else {
        let bestD = 500;
        for (const k in state.ramCache.slots) {
          const sl2 = state.ramCache.slots[k];
          if (sl2.evictedAt || sl2.removedAt) continue; // B10b: dead slots are not restored
          if (slotId != null && sl2.slot !== slotId) continue; // V6: a foreign slot does not pulse
          const d = Math.abs(sl2.tokens - snapshotTokens);
          if (d < bestD) { bestD = d; sl = sl2; }
        }
        if (sl) { sl.lastRestored = Date.now(); sl.restoreCount = (sl.restoreCount || 0) + 1; }
        else {
          // #8: the marker was erased (a better prompt was found) or was never created (the log started
          // mid-session) — the checkpoint REALLY exists in the server RAM (it was
          // just restored), the shelf must not lose it:
          // we recreate the marker from the restored line data. Guard against duplicates.
          const posMin = mPosMin ? parseInt(mPosMin[1], 10) : null;
          const posMax = mPosMax ? parseInt(mPosMax[1], 10) : null;
          const k = posMin != null && posMax != null
            ? (slotId != null ? slotId + ':' + posMin + ':' + posMax : posMin + ':' + posMax)
            : (slotId != null ? slotId + ':' + snapshotTokens : String(snapshotTokens));
          if (!state.ramCache.slots[k]) {
            const now = Date.now();
            state.ramCache.slots[k] = { key: k, slot: slotId, tokens: snapshotTokens, posMin, posMax, size: parseFloat(size), time: now, createdAt: now, lastRestored: now, restoreCount: 1 };
            recomputeCp();
          }
        }
      }
    }
  }
  else if (line.includes('forcing full prompt re-processing')) {
    if (isForeignTask(line)) return; // B1: a foreign task — do not reset the current state
    if (!state.task.id) return; // B1: the task boundary is not visible — no SYS event is created
    // Upstream semantics: the slot had a prefix match (LCP), but the KV
    // cannot be restored (SSM/Mamba hybrid, n_keep) — the server recomputes
    // the WHOLE prompt. The LCP match is a fact, we keep it; actually restored
    // is 0. cpSkew is guarded by kvReuse>0 (no phantom SKEW: the whole
    // prompt was recomputed, not the LCP "tail").
    state.task.kvRestoredTokens = 0;
    state.task.reprocessReason = 'FULL_REPROCESS';
    state.task.checkpointRestore = false;
    state.task.deltaTokens = state.task.totalPrompt;
    state.task.kvEventEmitted = true;
    if (state.task.lcpTokens > 0) {
      annotateSlotSelect(state.task.id, 'src_full_reprocess');
      addEvent('KV_RESTORE', 'FULL REPROCESS', 'ev_full_reproc_title', 'ev_full_reproc_detail',
        { lcp: state.task.lcpTokens.toLocaleString(), total: state.task.totalPrompt.toLocaleString() });
    } else {
      state.task.slotSource = 'empty';
      annotateSlotSelect(state.task.id, 'src_cold_start');
      addEvent('KV_RESTORE', 'KV MISS', 'ev_kv_miss_title', 'ev_kv_miss_detail',
        { total: state.task.totalPrompt.toLocaleString() });
    }
  }
  // 4b. In-Place VRAM hit (hot slot): "slot operator(): ... | cached n_tokens = 60730, memory_seq_rm [...]"
  // The server does NOT restore a checkpoint — the tensors are already in VRAM, the "restored context checkpoint" line is not written
  else if (line.includes('cached n_tokens =')) {
    if (isForeignTask(line)) return; // a foreign task (desync) — do not touch the current state
    if (!state.task.id) return; // the monitor attached mid-task — no boundary, no SYS event is created
    const m = line.match(/cached n_tokens\s*=\s*(\d+)/);
    if (m) {
      const tokens = parseInt(m[1], 10);
      // CANONICAL SEMANTICS (upstream server-context.cpp): "cached n_tokens = N"
      // only means "N tokens are already in the slot context" — NOT the KV source.
      // The source is determined ONLY by correlation with preceding events:
      //   LCP slot match          → slotSource = 'hot-vram'
      //   found better prompt     → slotSource = 'prompt-cache-ram'
      //   restored ctx checkpoint → checkpointRestore = true
      // Only the FIRST cached line after new prompt fixes the initial
      // state; subsequent values — progress of the current prefill.
      if (!state.task.cachedFirstSeen) {
        state.task.cachedFirstSeen = true;
        // E3: if a checkpoint was already lifted from RAM (restored before the first cached line),
        // the CACHED PREFIX twin is forbidden — the source is already fixed as checkpoint
        if (tokens > 0 && !state.task.checkpointRestore) {
          state.task.kvRestoredTokens = tokens; // cachedTokens N: length of the cached prefix
          state.task.deltaTokens = Math.max(0, state.task.totalPrompt - tokens);
          state.task.kvEventEmitted = true;
          const sourceKey = state.task.slotSource === 'prompt-cache-ram' ? 'src_prompt_cache'
                        : state.task.slotSource === 'hot-vram' ? 'src_hot_slot'
                        : 'src_unknown';
          annotateSlotSelect(state.task.id, sourceKey);
          addEvent('KV_RESTORE', 'CACHED PREFIX', 'ev_cached_prefix_title', 'ev_cached_prefix_detail',
            { tok: tokens.toLocaleString(), sourceKey });
        } else if (!state.task.kvEventEmitted) {
          // Explicit cold start: no cached prefix
          state.task.slotSource = state.task.slotSource || 'empty';
          state.task.kvEventEmitted = true;
          annotateSlotSelect(state.task.id, 'src_cold_start');
          addEvent('KV_RESTORE', 'KV MISS', 'ev_cold_start_title', 'ev_cold_start_detail', {});
        }
      }
    }
  }
  // 5. HOST RAM state (a rare authoritative snapshot).
  // "cache state: N prompts" — this is the PROMPT cache (server_prompt_cache.states,
  // one-shot snapshots of freed prompts), NOT checkpoints. Checkpoints —
  // a separate store (created/erased context checkpoint), the shelf
  // is maintained by them + resync (N of M) + sweep. "0 prompts" does NOT mean
  // "no checkpoints" (a real log: 123 of 124 lines "0 prompts" with live
  // checkpoints) — the slots reset on this line was removed (V5).
  else if (line.includes('cache state:')) {
    // N3: the limit may be negative (--cache-ram -1)
    // Upstream (server-task.cpp server_prompt_cache::update): «est» =
    // limit_tokens_cur = limit_size>0 ? max(limit_tokens, limit_size/size_per_token)
    // : limit_tokens — the EFFECTIVE token limit by which the server actually
    // evicts records. We show it, not the configured limit_tokens.
    const m = line.match(/cache state:\s*(\d+)\s*prompts,\s*([\d.]+)\s*MiB\s*\(limits:\s*(-?[\d.]+)\s*MiB(?:,\s*(\d+)\s*tokens)?(?:,\s*(\d+)\s*est)?/);
    if (m) {
      state.ramCache.promptsCount = parseInt(m[1], 10);
      state.ramCache.usedMiB = parseFloat(m[2]);
      state.ramCache.limitMiB = parseFloat(m[3]);
      state.ramCache.promptCacheMiB = parseFloat(m[2]);
      state.ramCache.limitTokens = m[4] ? parseInt(m[4], 10) : null;
      const est = m[5] ? parseInt(m[5], 10) : null;
      const conf = state.ramCache.limitTokens;
      // Effective = max(configured, est); without est (old format) = configured
      state.ramCache.limitTokensEff = (conf != null && est != null) ? Math.max(conf, est) : (est != null ? est : conf);
      state.ramCache.stateAt = new Date().toISOString();
      // V5: the checkpoint shelf reset on "0 prompts" was REMOVED — that is a different
      // store. The shelf lives by created/erased + resync + sweep.
      // Open the prompt cache record buffer: exactly promptsCount
      // record lines follow, commit by the counter (N2). At 0 — an immediate empty commit.
      cacheCollecting = [];
      cacheExpected = state.ramCache.promptsCount;
      cacheStartedAt = Date.now();
      if (cacheExpected === 0) { commitCacheEntries([]); cacheCollecting = null; }
      broadcastState();
    }
  }
  // 5b. Checkpoint shelf: exact key = slotId:pos_min:pos_max (H8).
  // Upstream (server-context.cpp) identifies a checkpoint by the object
  // {pos_min, pos_max, n_tokens, state} — EQUAL n_tokens are possible
  // (SWA window, different positions), so n_tokens by itself is NOT identity.
  // pos_min/pos_max are in ALL created/erased/restored lines — exact
  // match without fuzzy. Fallback (old builds without pos) — slotId:n_tokens.
  // The "N of M" index — the POSITION in THIS slot's list (renumbered after
  // erased), unsuitable as a key; M (cpMax) — the limit PER SLOT
  // (upstream: slot.prompt.checkpoints.size() >= n_ctx_checkpoints).
  else if (line.includes('created context checkpoint')) {
    sweepGone();
    const m = line.match(/created context checkpoint\s+(\d+)\s+of\s+(\d+)/);
    const mTok = line.match(/n_tokens\s*=\s*(\d+)/);
    const mSize = line.match(/size\s*=\s*([\d.]+)\s*MiB/);
    const mSlot = line.match(/\bid\s+(\d+)\s*\|/);
    const mPosMin = line.match(/pos_min\s*=\s*(\d+)/);
    const mPosMax = line.match(/pos_max\s*=\s*(\d+)/);
    if (m && mTok && mSize) {
      state.ramCache.cpCreated++;
      state.ramCache.cpMax = parseInt(m[2], 10);
      const tok = parseInt(mTok[1], 10);
      const slotId = mSlot ? parseInt(mSlot[1], 10) : null;
      const posMin = mPosMin ? parseInt(mPosMin[1], 10) : null;
      const posMax = mPosMax ? parseInt(mPosMax[1], 10) : null;
      const now = Date.now();
      const key = posMin != null && posMax != null
        ? (slotId != null ? slotId + ':' + posMin + ':' + posMax : posMin + ':' + posMax)
        : (slotId != null ? slotId + ':' + tok : String(tok));
      state.ramCache.slots[key] = { key, slot: slotId, tokens: tok, posMin, posMax, size: parseFloat(mSize[1]), time: now, createdAt: now, lastRestored: null, restoreCount: 0 };
      // Per-slot N of M reconciliation: the extra oldest of THIS slot are removed
      resyncCheckpoints(slotId, parseInt(m[1], 10), key);
      // KV SAVE RAM: the checkpoint is physically written to Host RAM — bind to the task of the line
      const mTask = line.match(/task\s+(\d+)/);
      addEvent('KV_RAM', 'KV SAVE RAM', 'ev_cp_save_title', 'ev_cp_save_detail',
        { n: m[1], m: m[2], tok: tok.toLocaleString(), size: mSize[1] }, mTask ? mTask[1] : null, false);
    }
  }
  else if (line.includes('erased invalidated context checkpoint') ||
           line.includes('erasing old context checkpoint') ||
           line.includes('erasing context checkpoint too close')) {
    sweepGone();
    state.ramCache.cpErased++;
    // H8: exact identity — pos_min/pos_max from the erased line (upstream
    // prints them in all erase lines). Fallback (old builds without pos):
    // n_tokens, and if absent pos_max + 1 — a heuristic (in upstream
    // n_tokens = pos_max + 1 is NOT an invariant: SWA window, update_pos),
    // then fuzzy ±200.
    const mTok = line.match(/n_tokens\s*=\s*(\d+)/);
    const mPosMin = line.match(/pos_min\s*=\s*(\d+)/);
    const mPosMax = line.match(/pos_max\s*=\s*(\d+)/);
    const evTokens = mTok ? parseInt(mTok[1], 10)
                 : (mPosMax ? parseInt(mPosMax[1], 10) + 1 : null);
    const mSize = line.match(/size\s*=\s*([\d.]+)\s*MiB/);
    const mSlot = line.match(/\bid\s+(\d+)\s*\|/);
    const slotId = mSlot ? parseInt(mSlot[1], 10) : null;
    // CHECKPOINT LOOP watchdog: N identical erase in a row (same task, slot,
    // pos_min/pos_max, line type). The event describes ONLY the OBSERVED
    // FACT (repetition of identical erase lines); we do not conclude livelock/stall
    // without state confirmation (is_processing) — only as a note.
    // The 60-second stall detector — a backstop; here we catch it in seconds.
    // The key includes taskId — the chain breaks on a task change.
    {
      const mTask = line.match(/task\s+(\d+)/);
      const taskId = mTask ? mTask[1] : null;
      const eraseKind = line.includes('erased invalidated') ? 'invalidated'
                       : line.includes('erasing old') ? 'old' : 'too-close';
      const loopKey = (taskId || '?') + ':' + (slotId != null ? slotId : '?') + ':'
                    + (mPosMin ? mPosMin[1] : '?') + ':' + (mPosMax ? mPosMax[1] : '?') + ':' + eraseKind;
      if (state.eraseLoop && state.eraseLoop.key === loopKey) {
        state.eraseLoop.count++;
      } else {
        state.eraseLoop = { key: loopKey, count: 1 };
      }
      if (state.eraseLoop.count === 5) {
        // #9: only the observed fact; stall signs — only if
        // confirmed by the state (is_processing)
        addEvent('KV_RAM', 'CHECKPOINT LOOP', 'ev_cp_loop_title',
          state.is_processing ? 'ev_cp_loop_detail_stall' : 'ev_cp_loop_detail',
          { tid: taskId || '?', posMin: mPosMin ? mPosMin[1] : '?', posMax: mPosMax ? mPosMax[1] : '?' },
          taskId, false);
      }
    }
    let evictSize = null;
    if (evTokens !== null) {
      const tok = evTokens;
      let key = null;
      if (mPosMin && mPosMax) {
        const k = slotId != null ? slotId + ':' + mPosMin[1] + ':' + mPosMax[1]
                                 : mPosMin[1] + ':' + mPosMax[1];
        key = state.ramCache.slots[k] ? k : null;
      }
      if (key === null) {
        const keyOf = t => slotId != null ? slotId + ':' + t : String(t);
        key = state.ramCache.slots[keyOf(tok)] ? keyOf(tok) : null;
      }
      if (key === null) {
        let bestD = 200;
        for (const k in state.ramCache.slots) {
          const sl = state.ramCache.slots[k];
          if (sl.evictedAt || sl.removedAt) continue;
          if (slotId != null && sl.slot !== slotId) continue; // V6: a foreign slot is not touched
          const d = Math.abs(sl.tokens - tok);
          if (d < bestD) { bestD = d; key = k; }
        }
      }
      // do not delete immediately: a red flash for 2 s, then swept
      if (key !== null) {
        state.ramCache.slots[key].evictedAt = Date.now();
        state.ramCache.slots[key].removedAt = Date.now();
        state.ramCache.slots[key].removeReason = line.includes('erased invalidated') ? 'invalidated' : 'capacity';
        evictSize = state.ramCache.slots[key].size;
      }
      recomputeCp(); // B3: a plain live count
    }
    // KV EVICT RAM: the checkpoint is physically erased from Host RAM
    const reasonKey = line.includes('erased invalidated') ? 'reason_invalidated'
               : line.includes('erasing old') ? 'reason_capacity'
               : 'reason_too_close';
    const sizeStr = mSize ? mSize[1] : (evictSize != null ? evictSize.toFixed(3) : '?');
    const mTask = line.match(/task\s+(\d+)/);
    addEvent('KV_RAM', 'KV EVICT RAM', 'ev_cp_evict_title', 'ev_cp_evict_detail',
      { tok: evTokens !== null ? evTokens.toLocaleString() : '?', size: sizeStr, reasonKey }, mTask ? mTask[1] : null, false);
  }
  // 5c. Superseding (upstream server-context.cpp create_checkpoint): a new
  // checkpoint at the SAME n_tokens point replaces the existing one — the old one is erased
  // (the line "superseding context checkpoint at n_tokens = N"). The shelf label
  // with this n_tokens is stale — we remove it (removedAt, reason superseded).
  else if (line.includes('superseding context checkpoint')) {
    sweepGone();
    const mTok = line.match(/n_tokens\s*=\s*(\d+)/);
    const mSlot = line.match(/\bid\s+(\d+)\s*\|/);
    const slotId = mSlot ? parseInt(mSlot[1], 10) : null;
    // Identity: pos_min/pos_max, if the line carries them (the current upstream
    // prints only n_tokens — fallback). Several checkpoints of one
    // slot may have the same n_tokens — we remove EXACTLY ONE
    // (the oldest by createdAt): underdefining is better than knocking out an extra one.
    const mPosMin = line.match(/pos_min\s*=\s*(\d+)/);
    const mPosMax = line.match(/pos_max\s*=\s*(\d+)/);
    let supTok = null;
    if (mTok) {
      supTok = parseInt(mTok[1], 10);
      const now = Date.now();
      let victim = null;
      for (const k in state.ramCache.slots) {
        const sl = state.ramCache.slots[k];
        if (sl.evictedAt || sl.removedAt) continue;
        if (slotId != null && sl.slot !== slotId) continue;
        const match = mPosMin && mPosMax
          ? (sl.posMin != null && sl.posMin === parseInt(mPosMin[1], 10) && sl.posMax === parseInt(mPosMax[1], 10))
          : (sl.tokens === supTok);
        if (!match) continue;
        if (!victim || (sl.createdAt || 0) < (victim.createdAt || 0)) victim = sl;
      }
      if (victim) {
        victim.removedAt = now;
        victim.removeReason = 'superseded';
      }
      recomputeCp();
    }
    const mTask = line.match(/task\s+(\d+)/);
    addEvent('KV_RAM', 'KV SUPERSEDE', 'ev_cp_supersede_title', 'ev_cp_supersede_detail',
      { tok: supTok !== null ? supTok.toLocaleString() : '?' }, mTask ? mTask[1] : null, false);
  }
  else if (line.includes('prompt_save:') && line.includes('saving prompt with length')) {
    const m = line.match(/saving prompt with length\s*(\d+),\s*total state size\s*=\s*([\d.]+)\s*MiB/);
    if (m) {
      // #7: a service line at a task boundary — NOT into the feed immediately (it would stick
      // to the previous task's card). We buffer it; on the next "new prompt"
      // it is flushed into the NEW task (see the INPUT branch). lifecycle=false: does not move
      // currentStage.
      pendingSysEvents.push({
        stage: 'PREFIX', tag: 'RAM CACHE', titleKey: 'ev_prompt_save_title',
        detailKey: 'ev_prompt_save_detail',
        params: { tok: Number(m[1]).toLocaleString(), size: m[2] }
      });
    }
  }
  // 6. PREFILL
  else if (line.includes('prompt processing') && line.includes('progress =')) {
    if (isForeignTask(line)) return; // a foreign task (desync) — do not touch the current state
    const m = line.match(/n_tokens\s*=\s*(\d+).*?progress\s*=\s*([\d.]+)(?:.*?([\d.]+)\s*tokens per second)?/);
    if (m) {
      state.currentStage = 'PREFILL';
      const pct = Math.round(parseFloat(m[2]) * 100);
      state.task.prefillProgress = pct;
      if (m[3]) { state.task.prefillTps = parseFloat(m[3]); state.lastPrefillTps = state.task.prefillTps; }
      
      const currentStep = Math.floor(pct / 25);
      if (currentStep > state.task.lastProgressStep && pct < 98) {
        state.task.lastProgressStep = currentStep;
        addEvent('PREFILL', 'PREFILL', 'ev_prefill_prog_title',
          m[3] ? 'ev_prefill_prog_detail_spd' : 'ev_prefill_prog_detail',
          { pct: currentStep * 25, tok: Number(m[1]).toLocaleString(), tps: m[3] });
      } else {
        broadcastState();
      }
    }
  } 
  // 7. SAMPLER — we do NOT publish "prefill finished" here:
  // the real token count on the GPU is known only from "prompt eval time" (branch 8)
  else if (line.includes('init sampler, took')) {
    state.currentStage = 'PREFILL';
    broadcastState();
  } 
  // 8. PREFILL TIMING
  else if (line.includes('prompt eval time =')) {
    const m = line.match(/prompt eval time\s*=\s*([\d.]+)\s*ms\s*\/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second/);
    if (m) {
      // Task binding: the line has a real task id — we ignore a desync from another task
      const tidM = line.match(/task\s+(\d+)/);
      if (tidM && state.task.id && tidM[1] !== state.task.id) {
        // A foreign task's timing (a fast request stream) — not applied to the current one
      } else {
        state.task.prefillMs = parseFloat(m[1]);
        state.task.prefillTokens = parseInt(m[2], 10);
        state.task.prefillTps = parseFloat(m[3]);
        state.lastPrefillTps = state.task.prefillTps; // M3: the final measurement overrides the intermediate one
        // #5: the "prompt eval time" line — the server's real measurement (always
        // >= 1 token, upstream estimates a minimum of 1) — seen=true
        state.task.prefillTimingSeen = true;

        const splitParams = { total: state.task.totalPrompt.toLocaleString(), reused: state.task.kvRestoredTokens.toLocaleString(), new: Number(m[2]).toLocaleString(), ms: m[1], tps: m[3] };
        if (!state.task.prefillReported) {
          state.task.prefillReported = true;
          addEvent('PREFILL', 'PREFILL', 'ev_prefill_done_title', 'ev_prefill_done_detail', splitParams);
        } else {
          // The Fallback (LCP-based estimate) was already published — replace it with the measured split
          const ev = state.events.find(e => e.stage === 'PREFILL' && e.tag === 'PREFILL' && e.titleKey === 'ev_prefill_done_title' && e.taskId === state.task.id);
          if (ev) { ev.detailKey = 'ev_prefill_done_measured'; ev.params = splitParams; state.eventsRev++; } // B2
          broadcastState();
        }
      }
    }
  }
  // 10. SPEC TOTAL (per-task): "draft acceptance" — ALL speculative implementations
  // (draft-mtp + ngram-mod together), the statistics are reset between tasks.
  else if (line.includes('draft acceptance =') || line.includes('draft acceptance rate =')) {
    const m = line.match(/draft acceptance(?:\s*rate)?\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*\/\s*(\d+)\s*generated\)(?:,\s*mean len\s*=\s*([\d.]+))?/);
    if (m) {
      // Honest binding: the task ID is taken from the line itself — the line arrives
      // AFTER DONE, and state.task.id may already be the next task.
      // There is NO isForeignTask guard here: a lagging line is not discarded,
      // it sticks to its OWN task in the history.
      const tidM = line.match(/task\s+(\d+)/);
      const targetTaskId = tidM ? tidM[1] : String(state.task.id || '');
      const isCurrent = targetTaskId !== '' && targetTaskId === String(state.task.id);
      if (isCurrent) {
        state.task.specAcceptance = (parseFloat(m[1]) * 100).toFixed(1);
        state.task.specMeanLen = m[4] ? parseFloat(m[4]) : 0;
        state.task.specAccepted = parseInt(m[2], 10);
        state.task.specGenerated = parseInt(m[3], 10);
      }
      const accPct = (parseFloat(m[1]) * 100).toFixed(1);
      // lifecycle=false: the line arrives AFTER DONE and must not move currentStage
      addEvent('MTP', 'MTP', 'ev_mtp_title', m[4] ? 'ev_mtp_detail_len' : 'ev_mtp_detail',
        { gen: m[3], acc: m[2], pct: accPct, len: m[4] }, targetTaskId || null, false);
      // The history record was already created: we fill in the SPEC fields by the task id from the line
      const h = (state.history || []).find(x => String(x.id) === targetTaskId);
      if (h) {
        h.specAccepted = parseInt(m[2], 10);
        h.specGenerated = parseInt(m[3], 10);
        h.specAcceptance = accPct;
        h.specMeanLen = m[4] ? parseFloat(m[4]) : 0;
        state.historyRev++; // B2: a history record change
        broadcastState();
      }
    }
  }
  // 10a. DRAFT-MTP — SESSION CUMULATIVE (common_speculative_print_stats:
  // counters live for the whole server lifetime, reset is not called).
  // Format: "statistics       draft-mtp: #calls(b,g,a) = X Y Z, #gen drafts = N,
  //          #acc drafts = N, #gen tokens = N, #acc tokens = N,
  //          #mean acc len = L, #acc rate/pos = (r1, r2, ...)"
  // The rate/pos length = the observed maximum position, NOT n_max.
  else if (line.match(/statistics\s+draft-mtp:/)) {
    const sm = state.specMtp;
    const mCalls = line.match(/#calls\(b,g,a\)\s*=\s*(\d+)\s+(\d+)\s+(\d+)/);
    const mGenD = line.match(/#gen drafts\s*=\s*(\d+)/);
    const mAccD = line.match(/#acc drafts\s*=\s*(\d+)/);
    const mGen = line.match(/#gen tokens\s*=\s*(\d+)/);
    const mAcc = line.match(/#acc tokens\s*=\s*(\d+)/);
    const mLen = line.match(/#mean acc len\s*=\s*([\d.]+)/);
    const mPos = line.match(/#acc rate\/pos\s*=\s*\(([-\d.,\s]+)\)/);
    if (mCalls) sm.calls = [parseInt(mCalls[1], 10), parseInt(mCalls[2], 10), parseInt(mCalls[3], 10)];
    if (mGenD) sm.genDrafts = parseInt(mGenD[1], 10);
    if (mAccD) sm.accDrafts = parseInt(mAccD[1], 10);
    if (mGen) sm.genTokens = parseInt(mGen[1], 10);
    if (mAcc) sm.accTokens = parseInt(mAcc[1], 10);
    if (mLen) sm.meanLen = parseFloat(mLen[1]);
    if (mPos) sm.posAcc = mPos[1].split(',').map(x => x.trim()).filter(Boolean);
    sm.invariantOk = specInvariantOk(sm.meanLen, sm.posAcc);
    broadcastState();
  }
  // 10b. Per-task "acc per pos" (SPEC TOTAL, all implementations): arrives right
  // after "draft acceptance" (TRC level), the task id — in the line prefix.
  // Upstream resizes the tuple to common_speculative_n_max (the real value —
  // specConfig.draftNMax from the startup log; for the MTP bars
  // state.specMtp.posAcc is used, not this line).
  else if ((line.includes('acc per pos =') || line.includes('acc rate/pos =')) && !line.match(/statistics\s+/)) {
    const m = line.match(/acc (?:per pos|rate\/pos)\s*=\s*\(([-\d.,\s]+)\)/);
    if (m) {
      const vals = m[1].split(',').map(x => x.trim()).filter(Boolean);
      if (vals.length > 0) {
        const rates = vals.map(x => parseFloat(x));
        const posPct = vals.map(x => (parseFloat(x) * 100).toFixed(0));
        const tidM = line.match(/task\s+(\d+)/);
        const targetTaskId = tidM ? tidM[1] : String(state.task.id || '');
        if (targetTaskId !== '' && targetTaskId === String(state.task.id)) {
          state.task.specPosAcc = posPct;
          state.task.specInvariantOk = specInvariantOk(state.task.specMeanLen, rates);
          broadcastState();
        }
        const h = (state.history || []).find(x => String(x.id) === targetTaskId);
        if (h) {
          h.specPosAcc = posPct;
          h.specInvariantOk = specInvariantOk(h.specMeanLen, rates);
          state.historyRev++; // B2: a history record change
          broadcastState();
        }
      }
    }
  }
  // 10b. CANCEL (the user interrupted the request)
  else if (line.includes('cancel task') && line.includes('id_task =')) {
    const m = line.match(/id_task\s*=\s*(\d+)/);
    if (m) {
      const tid = m[1];
      if (state.task.id != null && String(state.task.id) === tid) {
        addEvent('CANCEL', 'CANCEL', 'ev_cancel_title', 'ev_cancel_detail', { tid }, tid);
        pushHistory({ cancelled: true, isTruncated: false });
        state.currentStage = 'DONE';
        state._awaitingIdle = true;
        state._doneSince = Date.now();
        setTimeout(() => {
          if (state._awaitingIdle && !state.is_processing) {
            state._awaitingIdle = false;
            state.currentStage = 'IDLE';
            broadcastState();
          }
        }, 1500);
      } else {
        // cancelling an already finished/historical task — we mark the record
        const h = (state.history || []).find(x => String(x.id) === tid);
        if (h) { h.cancelled = true; state.historyRev++; broadcastState(); } // B2
      }
    }
  }
  // 10d. NGRAM-MOD — SESSION CUMULATIVE (the second speculation source).
  // Format: "statistics        ngram-mod:" — multiple spaces (logfmt), so the regex
  else if (line.match(/statistics\s+ngram-mod:/)) {
    const ng = state.ngram;
    const mCalls = line.match(/#calls\(b,g,a\)\s*=\s*(\d+)\s+(\d+)\s+(\d+)/);
    const mGenD = line.match(/#gen drafts\s*=\s*(\d+)/);
    const mAccD = line.match(/#acc drafts\s*=\s*(\d+)/);
    const mAcc = line.match(/#acc tokens\s*=\s*(\d+)/);
    const mGen = line.match(/#gen tokens\s*=\s*(\d+)/);
    const mLen = line.match(/#mean acc len\s*=\s*([\d.]+)/);
    const mPos = line.match(/#acc rate\/pos\s*=\s*\(([-\d.,\s]+)\)/);
    if (mCalls) ng.calls = [parseInt(mCalls[1], 10), parseInt(mCalls[2], 10), parseInt(mCalls[3], 10)];
    if (mGenD) ng.genDrafts = parseInt(mGenD[1], 10);
    if (mAccD) ng.accDrafts = parseInt(mAccD[1], 10);
    ng.accTokens = mAcc ? parseInt(mAcc[1], 10) : 0;
    ng.genTokens = mGen ? parseInt(mGen[1], 10) : 0;
    ng.meanLen = mLen ? parseFloat(mLen[1]) : 0;
    if (mPos) ng.posAcc = mPos[1].split(',').map(x => x.trim()).filter(Boolean);
    ng.invariantOk = specInvariantOk(ng.meanLen, ng.posAcc);
    broadcastState();
  }
  // 10b. CONTEXT SHIFT (upstream server-context.cpp): "slot context shift,
  // n_keep = N, n_left = N, n_discard = N" — the context was shifted, generation
  // continues. The truncated flag in stop processing will be = 1 (sticky),
  // but this is NOT an overflow — we mark it so the DONE branch does not shout OVERFLOW.
  else if (line.includes('slot context shift')) {
    if (!isForeignTask(line)) state.task.ctxShifted = true;
  }
  // 11. DONE
  else if (line.includes('stop processing:') && line.includes('truncated =')) {
    if (isForeignTask(line)) return; // protection against a foreign task overwriting the state
    const m = line.match(/truncated\s*=\s*(\d+)/);
    if (m && parseInt(m[1], 10) > 0) {
      state.task.isTruncated = true;
      // Upstream order: print_timings (eval time) BEFORE release (stop
      // processing) — the history line was already written in the eval-time branch, when
      // isTruncated was still false. We patch the flags so the SHIFTED/OVERFLOW
      // badge shows in the table (otherwise history is always isTruncated=false).
      const h = state.history.find(x => String(x.id) === String(state.task.id) && !x.cancelled);
      if (h) { h.isTruncated = true; h.ctxShifted = Boolean(state.task.ctxShifted); state.historyRev++; }
      if (state.task.ctxShifted) {
        addEvent('DONE', 'CONTEXT SHIFT', 'ev_ctx_shift_title', 'ev_ctx_shift_detail', {});
      } else {
        addEvent('DONE', 'OVERFLOW', 'ev_overflow_title', 'ev_overflow_detail', {});
      }
    }
    // The final token count in the slot (n_tokens from stop processing)
    const mN = line.match(/n_tokens\s*=\s*(\d+)/);
    if (mN) state.task.finalSlotTokens = parseInt(mN[1], 10);
  } 
  else if (line.includes('eval time =') && !line.includes('prompt eval')) {
    const m = line.match(/eval time\s*=\s*([\d.]+)\s*ms\s*\/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second/);
    if (m) {
      // Task binding: the line has a real task id — we ignore a desync from another task
      const tidM = line.match(/task\s+(\d+)/);
      if (tidM && state.task.id && tidM[1] !== state.task.id) {
        // A foreign task's final timing (a fast request stream) — not applied to the current one
      } else {
        state.task.decodeMs = parseFloat(m[1]);
        state.task.generatedTokens = parseInt(m[2], 10);
        state.task.decodeTps = parseFloat(m[3]);
        state.currentStage = 'DONE';
        state.is_processing = false;
        state._awaitingIdle = true;
        state._doneSince = Date.now();

        // A short answer (<3 s): the server does not send intermediate "n_gen =" lines
        // (the progress timer is 3 s), so the DECODE event was not published — we catch up
        if (!state.task.decodeStarted) {
          state.task.decodeStarted = true;
          addEvent('DECODE', 'DECODE', 'ev_decode_start_title', 'ev_decode_instant_detail', { tps: m[3] });
        }

        addEvent('DONE', 'DONE', 'ev_done_title',
          state.task.decodeTpsPeak ? 'ev_done_detail_peak' : 'ev_done_detail',
          { tok: m[2], sec: (parseFloat(m[1]) / 1000).toFixed(2), tps: m[3], peak: state.task.decodeTpsPeak });
        console.log(`\x1b[32m✔ [DONE]\x1b[0m Generated ${m[2]} tokens at ${m[3]} t/s`);

        pushHistory({});

        setTimeout(() => {
          if (state._awaitingIdle && !state.is_processing) {
            state._awaitingIdle = false;
            state.currentStage = 'IDLE';
            broadcastState();
          }
        }, 1500);
      }
    }
  }
}
if (rl) rl.on('line', (raw) => {
  if (LIVE_MODE) {
    // Live lines are written to disk ONLY in LIVE (the block inside if (LIVE_MODE);
    // in --play rl is not created at all). The label — the real time.
    // Exception — service idle spam: handleLine discards it anyway,
    // we do not bloat the disk
    if (!raw.includes('update_slots: all slots are idle'))
      try {
        const line = `[${new Date(realNow()).toISOString()}] ${raw}\n`;
        logStream.write(line);
        // #17: size accounting + rotation on exceeding the limit
        logSize += Buffer.byteLength(line);
        if (LOG_MAX_BYTES > 0 && logSize >= LOG_MAX_BYTES) rotateLog();
      } catch (_) {}
    // In PLAYBACK live lines are not parsed and not buffered
    if (state.mode !== 'LIVE') return;
  }
  handleLine(raw);
});

// A2: the single entry point of a line into the engine — replay, play, load-log.
// The "[ISO] " prefix gives currentLogTs (A3), then the clean line goes to handleLine.
function feedLogLine(raw) {
  if (!raw) return;
  const m = raw.match(/^\[([^\]]+)\] /);
  if (m) {
    const t = Date.parse(m[1]);
    if (!Number.isNaN(t)) currentLogTs = t;
  }
  handleLine(raw.replace(/^\[[^\]]+\] /, ''));
}

// A1: full engine reset (load-log / player restart). serverOnline and mode are not touched.
function resetAll() {
  // Revisions are strictly monotonic: SSE/C1 clients key on them — the next
  // values are computed BEFORE assign (the factory zeroes them).
  const nextEventsRev = state.eventsRev + 1;
  const nextHistoryRev = state.historyRev + 1;
  Object.assign(state, freshSessionState(), {
    eventsRev: nextEventsRev,
    historyRev: nextHistoryRev
  });
  pendingLcp.clear(); pendingRamLcp.clear(); pendingRamBase.clear(); pendingLru.clear(); lastSelectionSlot = null;
  cacheCollecting = null; cacheExpected = 0; lastStallWarnSec = 0; currentLogTs = null;
  pendingSysEvents.length = 0; // #7: service events do not survive an engine reset
  broadcastState();
}

// A1: the player (PLAY / load-log). Virtual clock: Date.now = targetWall + (realNow - wallAtLine),
// targetWall = anchorWall + (t - anchorLog)/speed. Pause freezes the clock,
// resume shifts anchorWall by the pause duration. A5: --max-gap compresses pauses.
const player = {
  lines: [], idx: 0, speed: PLAY_SPEED, paused: false, finished: false,
  lastT: null, lastDoneT: null, anchorLog: null, anchorWall: null, pausedAt: 0,
  lastTargetWall: null, wallAtLine: null, timer: null
};

function updatePlayerState() {
  state.player = { paused: player.paused, speed: player.speed, idx: player.idx, total: player.lines.length, finished: player.finished };
  broadcastState();
}

function playerFeed() {
  if (player.paused || player.finished) return;
  const lines = player.lines;
  while (player.idx < lines.length) {
    const l = lines[player.idx];
    const m = l.match(/^\[([^\]]+)\] /);
    const p = m ? Date.parse(m[1]) : NaN;
    if (!Number.isNaN(p)) player.lastT = p;
    if (player.lastT === null) { player.idx++; continue; }
    const t = player.lastT;
    if (player.anchorLog === null) { player.anchorLog = t; player.anchorWall = realNow(); }
    let dt = t - player.anchorLog;
    if (MAX_GAP_MS > 0 && dt > MAX_GAP_MS) dt = MAX_GAP_MS; // A5
    const targetWall = player.anchorWall + dt / player.speed;
    const nowWall = realNow();
    if (nowWall < targetWall) {
      player.timer = setTimeout(playerFeed, Math.min(targetWall - nowWall, 1000));
      return;
    }
    player.lastTargetWall = targetWall;
    player.wallAtLine = realNow();
    Date.now = () => targetWall + (realNow() - player.wallAtLine);
    feedLogLine(l);
    player.idx++;
    player.lastDoneT = t; // the label of the last PROCESSED line (for playerSetSpeed)
    player.anchorLog = t;
    player.anchorWall = targetWall;
    updatePlayerState();
  }
  playerFinish();
}

function playerFinish(silent = false) {
  player.finished = true;
  if (player.timer) { clearTimeout(player.timer); player.timer = null; }
  Date.now = realNow;
  if (!silent) console.log('\x1b[1m\x1b[32m🎬 Playback finished — server still running (final state)\x1b[0m');
  updatePlayerState();
}

// Return from PLAYBACK to LIVE: stop the player (without a log), the real clock,
// a full engine reset. n_ctx will be restored from /slots on the nearest poll.
function goLive() {
  playerFinish(true);
  resetAll();
  state.mode = 'LIVE';
  updatePlayerState();
}

function playerPause() {
  if (player.finished || player.paused) return;
  player.paused = true;
  if (player.timer) { clearTimeout(player.timer); player.timer = null; }
  player.pausedAt = realNow();
  // Freezing the virtual clock at the last target moment + the real time since the line
  const frozen = player.lastTargetWall != null
    ? player.lastTargetWall + (player.pausedAt - player.wallAtLine)
    : realNow();
  Date.now = () => frozen;
  updatePlayerState();
}

function playerResume() {
  if (!player.paused || player.finished) return;
  player.paused = false;
  const pauseDur = realNow() - player.pausedAt;
  if (player.anchorWall != null) player.anchorWall += pauseDur;
  player.pausedAt = 0;
  // Returning the virtual clock: the frozen moment (frozen) + the running real time
  player.lastTargetWall = player.lastTargetWall != null ? player.lastTargetWall + pauseDur : realNow();
  player.wallAtLine = realNow();
  Date.now = () => player.lastTargetWall + (realNow() - player.wallAtLine);
  playerFeed();
  updatePlayerState();
}

function playerSetSpeed(v) {
  const oldSpeed = player.speed;
  player.speed = Math.max(0.1, parseFloat(v) || 1);
  // Re-anchoring: the current log moment in the old scale, capped from above by
  // the label of the waiting line (otherwise it would fire on the nearest tick).
  // On pause the reference point — the pause moment: anchorWall is frozen until resume,
  // otherwise the pause time would count as played (and resume would shift it again)
  const ref = player.paused ? player.pausedAt : realNow();
  if (player.anchorLog != null) {
    let logNow = player.anchorLog + (ref - player.anchorWall) * oldSpeed;
    if (player.lastT != null && logNow > player.lastT) logNow = player.lastT;
    player.anchorLog = logNow;
    player.anchorWall = ref;
  }
  if (player.timer) { clearTimeout(player.timer); player.timer = null; }
  playerFeed();
  updatePlayerState();
}

function playerRestart() {
  if (player.lines.length === 0) return;
  if (player.timer) { clearTimeout(player.timer); player.timer = null; }
  player.idx = 0; player.lastT = null; player.lastDoneT = null;
  player.anchorLog = null; player.anchorWall = null;
  player.paused = false; player.finished = false; player.pausedAt = 0;
  player.lastTargetWall = null; player.wallAtLine = null;
  resetAll(); // full engine reset: lines are not played over the old events/history
  playerFeed();
  updatePlayerState();
}

function playerStart(lines) {
  if (player.timer) { clearTimeout(player.timer); player.timer = null; }
  player.lines = lines; player.idx = 0; player.lastT = null; player.lastDoneT = null;
  player.anchorLog = null; player.anchorWall = null;
  player.paused = false; player.finished = false; player.pausedAt = 0;
  player.lastTargetWall = null; player.wallAtLine = null;
  // #6: the speed is NOT reset — player.speed lives between load-log/restart
  // (4x → load-log → stays 4x); a reset would lose the user's choice
  playerFeed();
  updatePlayerState();
}

let serverPollBusy = false;
let lastStallWarnSec = 0;

// /metrics llama.cpp (Prometheus format): parsing without allocations — line by line,
// only the 4 counters of interest. Called only in LIVE (pollServer).
function parsePrometheusMetrics(text) {
  if (!text) return;
  const m = state.serverMetrics;
  let pos = 0;
  const len = text.length;
  while (pos < len) {
    let nl = text.indexOf('\n', pos);
    if (nl < 0) nl = len;
    if (text.charCodeAt(pos) !== 35) { // '#' — a comment/HELP/TYPE
      const sp = text.indexOf(' ', pos);
      if (sp > pos && sp < nl) {
        let key = text.slice(pos, sp);
        const br = key.indexOf('{');
        if (br >= 0) key = key.slice(0, br); // llamacpp:foo{slot="0"} -> llamacpp:foo
        const valEnd = text.charCodeAt(nl - 1) === 13 ? nl - 1 : nl; // \r\n
        const v = parseFloat(text.slice(sp + 1, valEnd));
        if (Number.isFinite(v)) {
          switch (key) {
            case 'llamacpp:requests_deferred':      m.queueDeferred = v; break;
            case 'llamacpp:prompt_tokens_total':    m.promptTokensTotal = v; break;
            case 'llamacpp:tokens_predicted_total': m.tokensPredictedTotal = v; break;
          }
        }
      }
    }
    pos = nl + 1;
  }
  m.fetchedAt = Date.now();
}

async function pollServer() {
  // PLAYBACK: do not poll the server — is_processing/n_prompt_tokens/generatedTokens
  // must not overwrite the playback state (serverOnline is not touched)
  if (state.mode !== 'LIVE') return;
  if (serverPollBusy) return;
  serverPollBusy = true;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 400);
    const res = await fetch(`${SERVER_URL}/slots`, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (Array.isArray(data) && data.length > 0) {
      // C1/C2/C5: id_task — the ONLY authoritative key of the current task.
      // A stale HTTP response of another task (arrived after a task change) — DISCARD:
      // it must not overwrite the active task's counters.
      // Without a tracked task (id == null) the slot is used only for
      // slot-level fields (occupancy/is_processing/n_ctx) — task counters
      // are not applied (the "new prompt" line will fill them).
      const activeId = state.task.id;
      const slot = activeId != null
        ? data.find(s => s.id_task != null && String(s.id_task) === String(activeId))
        : (data.find(s => s.is_processing) || data[0]);
      const prevOnline = state.serverOnline;
      const prevUsed = state.n_prompt_tokens;
      const prevProc = state.is_processing;
      const prevDecoded = state.task.generatedTokens;

      state.serverOnline = true;
      if (slot) {
        if (slot.n_ctx) confirmNctx(slot.n_ctx, 'slots'); // V1 cross-check: /slots vs log lines
        state.n_prompt_tokens = slot.n_prompt_tokens || 0;
        state.is_processing = Boolean(slot.is_processing);

        if (activeId != null) {
          // The four authoritative to_json fields (llama.cpp server-context.cpp):
          // n_prompt_tokens_cache     → KV REUSE (n_past; 0 = no reuse)
          // n_prompt_tokens_processed → PREFILL (0 — measured, not an estimate;
          // a full KV hit gives 1, not 0: n_past-- in TAG_PROMPT_LOGITS)
          // next_token[0].n_decoded   → DECODE (slot.n_decoded does not exist in upstream)
          // n_prompt_cached is written by upstream EXACTLY ONCE per task
          // (server-context.cpp: slot.stats.n_prompt_cached = n_past) — within
          // a task the value is constant. If the log already proved the KV source
          // (kvEventEmitted: restored checkpoint / full re-processing /
          // cached-first), /slots carries the same n_past (±1 from TAG_PROMPT_LOGITS)
          // and is redundant — and a stale in-flight response (the slot got the task BEFORE
          // prompt-processing, the stats are still of the previous task) is able
          // to overwrite the proven value (94676 → 0 or 0 → 94676).
          if (Number.isFinite(slot.n_prompt_tokens_cache) && !state.task.kvEventEmitted) {
            state.task.kvRestoredTokens = slot.n_prompt_tokens_cache;
          }
          // n_prompt_processed: reset to 0 at task start, then only
          // increments (a context shift does not touch the stats) — within a task
          // non-decreasing. A stale in-flight snapshot can only be LOWER.
          if (Number.isFinite(slot.n_prompt_tokens_processed)) {
            state.task.prefillTokens = Math.max(state.task.prefillTokens, slot.n_prompt_tokens_processed);
            // #5: seen=true only on a POSITIVE measured value. 0 from
            // /slots is not proof of measurement (the slot may not have started
            // prompt-processing yet); a legitimate measured 0 (context shift)
            // comes from the "prompt eval time" line.
            if (slot.n_prompt_tokens_processed > 0) state.task.prefillTimingSeen = true;
          }
          // Smooth update of the decode counter from /slots (the value only,
          // no influence on currentStage — the lifecycle is driven
          // exclusively by the stdin lines)
          const nt = Array.isArray(slot.next_token) && slot.next_token.length > 0 ? slot.next_token[0] : null;
          if (nt && Number.isFinite(nt.n_decoded) && nt.n_decoded > state.task.generatedTokens) {
            state.task.generatedTokens = nt.n_decoded;
          }
        }
      }

      // The long-prefill detector
      if (state.is_processing && state._processingStart > 0) {
        const elapsedSec = Math.round((Date.now() - state._processingStart) / 1000);
        // Debounce protection: pollServer walks every 500 ms, the same second
        // could have hit the check twice
        if (elapsedSec > 60 && elapsedSec % 60 === 0 && state.currentStage === 'PREFILL' && elapsedSec !== lastStallWarnSec) {
          lastStallWarnSec = elapsedSec;
          addEvent('PREFILL', 'STALL_WARN', 'ev_stall_title', 'ev_stall_detail', { sec: elapsedSec });
        }
      }

      if (!prevOnline || prevUsed !== state.n_prompt_tokens || prevProc !== state.is_processing || prevDecoded !== state.task.generatedTokens) {
        broadcastState();
      }
    } else {
      markOffline();
    }
  } catch (_) {
    markOffline();
  } finally {
    serverPollBusy = false;
  }
}
// /metrics — its OWN interval (2 s) and its OWN busy flag: a slow /metrics
// cannot delay /slots (500 ms) and vice versa. An unreceived /metrics is not
// fatal — the server may be built without --metrics, the status comes from /slots.
let metricsPollBusy = false;
async function pollMetrics() {
  if (state.mode !== 'LIVE' || metricsPollBusy) return;
  metricsPollBusy = true;
  try {
    if (state.serverOnline) {
      const mCtrl = new AbortController();
      const mTimer = setTimeout(() => mCtrl.abort(), 500);
      const mRes = await fetch(`${SERVER_URL}/metrics`, { signal: mCtrl.signal });
      clearTimeout(mTimer);
      if (mRes.ok) {
        const pm = state.serverMetrics;
        const prevQ = pm.queueDeferred;
        const prevP = pm.promptTokensTotal, prevT = pm.tokensPredictedTotal;
        const firstSeen = !pm.metricsSeen; // #18: the first /metrics response
        pm.metricsSeen = true;
        parsePrometheusMetrics(await mRes.text());
        if (firstSeen || prevQ !== pm.queueDeferred ||
            prevP !== pm.promptTokensTotal || prevT !== pm.tokensPredictedTotal) {
          broadcastState();
        }
      }
    }
  } catch (_) { /* /metrics unavailable — ignore */ }
  finally { metricsPollBusy = false; }
}
if (LIVE_MODE) {
  setInterval(pollServer, 500);
  pollServer();
  setInterval(pollMetrics, 2000);
  pollMetrics();
}

// R4: effect time constants (shared by the server and the client — injected into
// the page via JSON.stringify/toString).
const FX = { born: 900, touch: 8000, death: 2000, beam: 1500, grow: 500 };
const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;
const decay = (age, ms) => (age < 0 || age > ms) ? 0 : (1 - age / ms) ** 2;
// The effect state of an entity at the moment now (pure: only timestamps).
function fx(e, now) {
  return {
    born:  e.createdAt ? decay(now - e.createdAt, FX.born) : 0,
    touch: e.touchedAt ? decay(now - e.touchedAt, FX.touch) : 0,
    death: e.removedAt ? clamp01((now - e.removedAt) / FX.death) : 0,
    until: Math.max(e.createdAt ? e.createdAt + FX.born : 0,
                    e.touchedAt ? e.touchedAt + FX.touch : 0,
                    e.removedAt ? e.removedAt + FX.death : 0)
  };
}

// A pure function: a state snapshot -> the rings model. No DOM, no Date.now —
// it can be computed on the server (--ring-model) and in the browser (injection via
// toString). Ring 2 — checkpoints (the tokens/n_ctx axis), ring 3 — prompt
// cache (the MiB/limitMiB axis). until = the end of the last effect (R4).
function buildRingModel(s) {
  const ram = s.ramCache || {};
  // V1: no fallback — we do not build labels without an axis. n_ctxMismatch → fail-closed:
  // the sources diverged, the scale is knowingly wrong — we annul the axis (frac = null).
  const nCtx = s.n_ctxMismatch ? null : (Number(s.n_ctx) || null);

  // Ring 2: all checkpoints — live + dying (the 2 s window: evictedAt/removedAt
  // is swept by sweepGone). The dying ones carry removedAt — the client draws them
  // as a red fade with an outward drift.
  const cpMarks = Object.values(ram.slots || {})
    .map(sl => ({
      slot: sl.slot != null ? sl.slot : null, // V6: checkpoint ownership (per-slot)
      tokens: sl.tokens,
      sizeMiB: sl.size,
      frac: nCtx ? sl.tokens / nCtx : null, // the axis share (0..1); angle = START + frac*2π; null = the axis is unknown
      createdAt: sl.createdAt || sl.time || null,
      touchedAt: sl.lastRestored || null,
      removedAt: sl.removedAt || sl.evictedAt || null,
      restoreCount: sl.restoreCount || 0,
      removeReason: sl.removeReason || null
    }))
    .sort((a, b) => a.tokens - b.tokens);
  const liveCpCount = cpMarks.filter(m => !m.removedAt).length;
  const cp = {
    marks: cpMarks,
    count: ram.cpCurrent != null ? ram.cpCurrent : liveCpCount,
    max: ram.cpMax,
    state: ram.cpMax == null ? 'na' : 'ok'
  };

  // Ring 3: used = the authoritative promptCacheMiB (from cache state), limit =
  // limitMiB. The model returns an ordered list of records (live + dying)
  // and the scale k: the painter lays out the arcs in order, the dying one shrinks
  // in place, the following ones slide into the freed spot (B2). The sum of the
  // live arcs' lengths = used/limit (the R3 gate).
  const limit = ram.limitMiB != null ? Number(ram.limitMiB) : null;
  const used = ram.promptCacheMiB != null ? Number(ram.promptCacheMiB) : null;
  // N3: without a limit (limit <= 0) the full ring = used; state 'unlimited'
  const limitEff = (limit != null && limit > 0) ? limit : (used != null && used > 0 ? used : null);
  const ramState = limit == null ? 'na' : (limit > 0 ? 'ok' : 'unlimited');
  const allEntries = ram.entries || [];
  const liveEntries = allEntries.filter(e => !e.removedAt);
  const sumLive = liveEntries.reduce((a, e) => a + (Number(e.sizeMiB) || 0), 0);
  const items = allEntries.map(e => ({ key: e.key, tokens: e.tokens, checkpoints: e.checkpoints,
    sizeMiB: Number(e.sizeMiB) || 0, createdAt: e.createdAt, touchedAt: e.touchedAt, removedAt: e.removedAt }));
  const k = (used != null && sumLive > 0) ? used / sumLive : 1;
  if (used != null && used > 0 && sumLive === 0) { // a build without per-line records
    items.push({ key: '~', tokens: 0, checkpoints: 0, sizeMiB: used, createdAt: null, touchedAt: null, removedAt: null });
  }
  const tokensUsed = liveEntries.reduce((a, e) => a + (Number(e.tokens) || 0), 0);
  const ramRing = {
    items: items,
    k: k,
    used: used,
    limit: limitEff, // the effective limit for scaling (N3)
    tokensUsed: tokensUsed,
    limitTokens: ram.limitTokens != null ? ram.limitTokens : null,
    limitTokensEff: ram.limitTokensEff != null ? ram.limitTokensEff : null,
    state: ramState
  };

  // until = the end of the last effect (R4): the controller spins the frame loop
  // while now < until, and stops in idle.
  let until = 0;
  for (const m of cpMarks) {
    const u = Math.max(m.createdAt ? m.createdAt + FX.born : 0,
                       m.touchedAt ? m.touchedAt + FX.touch : 0,
                       m.removedAt ? m.removedAt + FX.death : 0);
    if (u > until) until = u;
  }
  for (const it of items) {
    const u = Math.max(it.createdAt ? it.createdAt + FX.grow + FX.born : 0,
                       it.touchedAt ? it.touchedAt + FX.touch : 0,
                       it.removedAt ? it.removedAt + FX.death : 0);
    if (u > until) until = u;
  }

  return { cp: cp, ram: ramRing, until: until };
}

// A pure function: the ring-3 layout at the moment now (B2). A dying arc
// shrinks in place (1 - death), the following arcs slide into the freed
// spot — the intervals do not overlap. No DOM/ctx: verified by tests.
// Returns { arcs, until }: arcs — all drawn arcs (live + dying),
// until — the end of the last effect.
function computeRing3Layout(ram, now, fxEnabled) {
  const arcs = [];
  let until = 0, total = 0;
  if (ram.limit == null || ram.limit <= 0) return { arcs: arcs, until: 0 };
  for (const it of ram.items) {
    const death = it.removedAt ? clamp01((now - it.removedAt) / FX.death) : 0;
    if (it.removedAt && (!fxEnabled || death >= 1)) continue; // the gap is fully closed
    const grow = (fxEnabled && it.createdAt) ? clamp01((now - it.createdAt) / FX.grow) : 1;
    const len = it.sizeMiB * ram.k / ram.limit * (it.removedAt ? 1 - death : grow);
    arcs.push({ key: it.key, tokens: it.tokens, checkpoints: it.checkpoints, sizeMiB: it.sizeMiB,
               start: 0, len: len, createdAt: it.createdAt, touchedAt: it.touchedAt,
               removedAt: it.removedAt, death: death, grow: grow });
    total += len;
    const u = Math.max(it.createdAt ? it.createdAt + FX.grow + FX.born : 0,
                       it.touchedAt ? it.touchedAt + FX.touch : 0,
                       it.removedAt ? it.removedAt + FX.death : 0);
    if (u > until) until = u;
  }
  // N1: the dying arc occupies its spot while shrinking, and the live ones already occupy
  // used/limit — at the limit the sum may exceed 1 and the tail wraps onto the start.
  // A second pass scales all arcs so the sum does not exceed 1.
  const scale = total > 1 ? 1 / total : 1;
  let acc = 0;
  for (const a of arcs) { a.len *= scale; a.start = acc; acc += a.len; }
  return { arcs: arcs, until: until };
}

const HTML_PAGE = `
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Llama.cpp Canonical Engine Monitor</title>
<style>
:root{
  --bg:#06090e;--panel:#0d131d;--panel2:#090e15;--border:#1a2436;--text:#f1f5f9;--dim:#64748b;
  --input:#38bdf8;--prefix:#6366f1;--slot:#94a3b8;--reuse:#22c55e;--kv-save:#14b8a6;--kv-evict:#475569;--prefill:#eab308;--decode:#a855f7;--mtp:#06b6d4;--mtp-dim:#22d3ee;--red:#ef4444;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;padding:12px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,monospace;background:var(--bg);color:var(--text);display:flex;flex-direction:column;overflow:hidden}
.badge{padding:3px 9px;border-radius:6px;font-size:11.5px;font-weight:bold}.badge-on{background:#15803d;color:#fff}.badge-off{background:#b91c1c;color:#fff}.badge-warn{background:#dc2626;color:#fff;animation:blink 1s infinite}@keyframes blink{50%{opacity:.5}}
.tag-pill{padding:2px 6px;border-radius:4px;font-weight:bold;font-size:10.5px}
.flow-diagram{margin:0;padding:0;display:flex;align-items:center;justify-content:center}
.flow-diagram svg{max-height:42px;width:auto}
.flow-overlay{position:fixed;inset:0;z-index:1000;background:rgba(6,9,14,.9);display:none;flex-direction:column}
.flow-overlay.open{display:flex}
.flow-viewer-toolbar{display:flex;align-items:center;gap:8px;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--border);flex-shrink:0}
.flow-viewer-title{font-size:11px;font-weight:700;color:var(--dim);letter-spacing:0.5px;margin-right:auto}
.flow-viewer-toolbar button{background:var(--panel2);border:1px solid var(--border);color:var(--text);font-family:inherit;font-size:12px;font-weight:700;padding:3px 10px;border-radius:5px;cursor:pointer}
.flow-viewer-toolbar button:hover{border-color:var(--input);color:var(--input)}
#fv-zoom-val{font-size:11px;color:var(--dim);min-width:44px;text-align:center}
.flow-viewer-canvas{flex:1;position:relative;overflow:hidden;cursor:grab}
.flow-viewer-canvas.dragging{cursor:grabbing}
.flow-viewer-canvas svg{position:absolute;top:0;left:0;transform-origin:0 0;will-change:transform;max-width:none}
.flow-viewer-canvas svg foreignObject{overflow:visible !important}
svg .nodeLabel{font-size:10px !important;line-height:1.15 !important;color:#f1f5f9 !important;text-align:center !important;display:block !important}
svg .nodeLabel small{font-size:8px !important;line-height:1 !important;color:#94a3b8 !important;display:block !important;margin-top:3px !important}
.metrics-guide-drawer{background:#090e16;border-top:1px solid var(--border);padding:14px 18px;max-height:45vh;overflow-y:auto;display:block;transition:all 0.3s ease}
.metrics-guide-drawer.collapsed{display:none}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}
.metric-card{background:#0d1522;border:1px solid #1a2538;border-radius:6px;padding:10px 12px;display:flex;flex-direction:column;gap:5px;font-family:-apple-system,BlinkMacSystemFont,monospace}
.metric-head{display:flex;align-items:center;gap:8px;font-size:11.5px}
.m-pill{font-size:9.5px;font-weight:bold;padding:1px 6px;border-radius:4px;border:1px solid;background:rgba(255,255,255,0.04)}
.m-desc{margin:0;font-size:11px;color:#cbd5e1;line-height:1.35}
.m-code{background:#05080d;border:1px solid #162030;color:var(--input);font-size:10px;padding:3px 6px;border-radius:4px;font-family:monospace;margin:2px 0}
.m-why{font-size:10px;color:var(--dim);line-height:1.3}
.m-why b{color:#94a3b8}
/* The 📐 LEGEND button in the history table header */
.btn-legend {
  background: #151e2b;
  border: 1px solid var(--border);
  color: var(--input);
  font: bold 10px -apple-system, BlinkMacSystemFont, monospace;
  padding: 2px 7px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.btn-legend:hover {
  border-color: var(--input);
  box-shadow: 0 0 8px rgba(56,189,248,0.25);
  color: #fff;
}

.main-layout { flex: 1; min-height: 0; display: grid; grid-template-columns: 1.3fr 0.9fr; gap: 12px; }
.left-col { display: flex; flex-direction: column; gap: 10px; min-height: 0; height: 100%; }
.right-col { display: flex; flex-direction: column; min-height: 0; height: 100%; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 9px; padding: 12px; position: relative; }
.card-history { flex: 1; min-height: 120px; display: flex; flex-direction: column; }
.card-log { flex: 1; min-height: 0; height: 100%; display: flex; flex-direction: column; }
.main-layout.log-collapsed { grid-template-columns: 1fr 36px; }
.main-layout.log-collapsed .card-log { padding: 4px; }
.main-layout.log-collapsed .card-log > *:not(.card-log-collapsed-bar) { display: none; }
.card-log-collapsed-bar { display: none; flex-direction: column; align-items: center; gap: 8px; padding: 8px 0; cursor: pointer; }
.main-layout.log-collapsed .card-log-collapsed-bar { display: flex; }
.card-log-collapsed-bar .log-bar-label { writing-mode: vertical-rl; font-size: 11px; font-weight: bold; letter-spacing: 1px; color: var(--dim); }
.card-log-collapsed-bar .log-bar-badge { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid #64748b; border-radius: 8px; font-size: 10px; padding: 2px 5px; font-variant-numeric: tabular-nums; }
.donut-layout{display:flex;align-items:center;gap:18px}.donut-box{position:relative;width:370px;height:370px;flex-shrink:0}.donut-box canvas{width:370px;height:370px;display:block}
.ring-tooltip{position:absolute;z-index:10;pointer-events:none;display:none;background:#0d1522;border:1px solid var(--border);border-radius:6px;padding:5px 8px;font-size:11px;line-height:1.5;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.5)}
.ring-tooltip .tt-dim{color:var(--dim);font-size:10px}
.donut-core { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; border-radius: 50%; pointer-events: none; overflow: hidden; }
.donut-core::after { content: ""; position: absolute; inset: 0; border-radius: 50%; background: radial-gradient(circle, rgba(56,189,248,.35) 0%, transparent 70%); opacity: 0; pointer-events: none; transition: opacity .4s ease; }
.donut-core.breathe-idle::after { animation: breatheGlow 3.2s ease-in-out infinite; }
@keyframes breatheGlow { 0%, 100% { opacity: 0.15; transform: scale(0.95); } 50% { opacity: 0.85; transform: scale(1.05); } }
.phase-badge{font-size:12px;font-weight:bold;padding:3px 14px;border-radius:20px;background:#151e2b;color:var(--text);border:1px solid var(--border);margin-bottom:3px;transition:all .25s ease}
.live-speed{font-size:26px; font-weight:bold; color:#fff; margin:3px 0; white-space: nowrap;}
.live-pct{font-size:12px;color:var(--dim)}
.phase-IDLE { box-shadow: 0 0 0 1px rgba(100,116,139,.4), 0 0 16px rgba(56,189,248,.25); }
.phase-INPUT { box-shadow: 0 0 0 1px var(--input), 0 0 20px rgba(56,189,248,.55); color: var(--input); }
.phase-PREFIX { box-shadow: 0 0 0 1px var(--prefix), 0 0 20px rgba(99,102,241,.55); color: var(--prefix); }
.phase-KV_RESTORE { box-shadow: 0 0 0 1px var(--reuse), 0 0 20px rgba(34,197,94,.55); color: var(--reuse); }
.phase-PREFILL { box-shadow: 0 0 0 1px var(--prefill), 0 0 20px rgba(234,179,8,.55); color: var(--prefill); }
.phase-DECODE { box-shadow: 0 0 0 1px var(--decode), 0 0 20px rgba(168,85,247,.6); color: var(--decode); }
.phase-MTP { box-shadow: 0 0 0 1px var(--mtp), 0 0 20px rgba(6,182,212,.55); color: var(--mtp); }
.phase-DONE { box-shadow: 0 0 0 1px #64748b, 0 0 12px rgba(100,116,139,0.25); color: #94a3b8; }
.donut-legend{flex:1;display:flex;flex-direction:column;gap:5px}.legend-row{display:flex;justify-content:space-between;align-items:center;background:#090e15;padding:5px 10px;border-radius:5px;border:1px solid var(--border);transition:all .25s ease}.legend-row.active-row{border-color:var(--input);background:#121b28}.legend-title{display:flex;align-items:center;gap:6px;font-size:11.5px;font-weight:bold}.dot{width:8px;height:8px;border-radius:2px}.legend-val{font-size:12.5px;font-weight:bold;color:#fff}
.bar-bg{display:flex;height:14px;border-radius:4px;overflow:hidden;background:#16202f;border:1px solid var(--border);margin:4px 0}.bar-seg{height:100%;transition:width .35s ease}
.mtp-grid{display:flex;gap:4px;margin-top:4px}.mtp-item{flex:1 1 auto;text-align:center;font-size:9px}
.mtp-col{background:#0b111a;height:28px;display:flex;align-items:flex-end;border-radius:3px;overflow:hidden;border:1px solid #1a2538}
.mtp-fill{width:100%;background:rgba(6,182,212,0.25);border-top:2px solid var(--mtp-dim);box-shadow:0 0 6px rgba(6,182,212,0.2);transition:height .35s cubic-bezier(0.4, 0, 0.2, 1)}
.hist-row-live{background:rgba(56,189,248,0.08) !important}
.hist-row-live td:first-child{box-shadow:inset 3px 0 0 var(--input) !important}
.log-box { flex: 1; height: 100%; min-height: 0; overflow-y: auto; overflow-anchor: none; background: #04070b; border: 1px solid var(--border); border-radius: 6px; padding: 8px; font-family: monospace; font-size: 12px; }
.task-card { background: #090e16; border: 1px solid #1a2538; border-radius: 7px; margin-bottom: 8px; overflow: hidden; transition: all 0.25s ease; }
.task-card.active-task-card { border-color: var(--input); box-shadow: 0 0 12px rgba(56,189,248,0.22); }
.task-card.highlight-pulse { border-color: var(--input) !important; box-shadow: 0 0 18px var(--input) !important; animation: taskPulse 1.2s ease-out 1; }
@keyframes taskPulse { 0% { transform: scale(0.985); } 50% { transform: scale(1.01); } 100% { transform: scale(1); } }
.task-card-header { display: flex; justify-content: space-between; align-items: center; background: #0d1522; padding: 6px 10px; cursor: pointer; border-bottom: 1px solid #162030; font-size: 11.5px; }
.task-card-header:hover { background: #131f30; }
.task-card-title { display: flex; align-items: center; gap: 8px; font-weight: bold; }
.task-badge-id { background: #1a273b; color: var(--input); padding: 2px 7px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(56,189,248,0.3); }
.task-card-summary { font-size: 11px; color: var(--dim); }
.task-card-body { padding: 4px 6px; display: flex; flex-direction: column; gap: 3px; }
.log-row { margin: 2px 0; padding: 4px 8px; border-radius: 4px; background: #070b12; border-left: 3px solid var(--border); display: flex; align-items: flex-start; gap: 8px; line-height: 1.3; font-size: 10.5px; }
.log-tag { width: 104px; min-width: 104px; height: 22px; flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center; box-sizing: border-box; font-size: 9.5px; padding: 0 4px; border-radius: 4px; font-weight: bold; letter-spacing: 0.2px; margin-top: 1px; }
.log-msg { flex: 1; min-width: 0; word-break: break-word; color: var(--dim) !important; }
.log-msg b { color: var(--dim) !important; font-weight: 600; }
.log-row .log-time { display: none; width: 82px; min-width: 82px; flex-shrink: 0; color: var(--dim); font-variant-numeric: tabular-nums; font-size: 10px; }
#log-feed.show-time .log-row .log-time { display: inline-block; }
#log-feed.hide-INPUT .log-row.INPUT { display: none !important; }
#log-feed.hide-PREFIX .log-row.PREFIX:not([data-tag="SLOT SELECT"]) { display: none !important; }
#log-feed.hide-SLOT_SELECT .log-row[data-tag="SLOT SELECT"] { display: none !important; }
#log-feed.hide-KV_RESTORE .log-row.KV_RESTORE { display: none !important; }
#log-feed.hide-KV_RAM .log-row.KV_RAM { display: none !important; }
#log-feed.hide-PREFILL .log-row.PREFILL { display: none !important; }
#log-feed.hide-DECODE .log-row.DECODE { display: none !important; }
#log-feed.hide-MTP .log-row.MTP { display: none !important; }
#log-feed.hide-DONE .log-row.DONE { display: none !important; }
.log-row.INPUT { border-left-color: var(--input); }
.log-row.INPUT .log-tag { background: rgba(56,189,248,.15); color: var(--input); border: 1px solid var(--input); }
.log-row.PREFIX { border-left-color: var(--prefix); }
.log-row.PREFIX .log-tag { background: rgba(99,102,241,.15); color: var(--prefix); border: 1px solid var(--prefix); }
.log-row.PREFIX[data-tag="SLOT SELECT"] { border-left-color: var(--slot); }
.log-row.PREFIX[data-tag="SLOT SELECT"] .log-tag { background: rgba(148,163,184,.12); color: var(--slot); border: 1px solid var(--slot); }
.log-row.MTP { border-left-color: var(--mtp); }
.log-row.MTP .log-tag { background: rgba(6,182,212,0.12); color: var(--mtp-dim); border: 1px solid rgba(6,182,212,0.4); }
.log-row.KV_RESTORE { border-left-color: var(--reuse); }
.log-row.KV_RESTORE .log-tag { background: rgba(34,197,94,.15); color: var(--reuse); border: 1px solid var(--reuse); }
.log-row.KV_RESTORE[data-tag="KV REUSE RAM"] { border-left-color: var(--kv-save); }
.log-row.KV_RESTORE[data-tag="KV REUSE RAM"] .log-tag { background: rgba(20,184,166,.15); color: var(--kv-save); border-color: var(--kv-save); }
.log-row.KV_RESTORE[data-tag="CACHED PREFIX"] { border-left-color: var(--reuse); }
.log-row.KV_RESTORE[data-tag="CACHED PREFIX"] .log-tag { background: rgba(34,197,94,.15); color: var(--reuse); border-color: var(--reuse); }
.log-row.KV_RESTORE[data-tag="KV MISS"] { border-left-color: var(--red); }
.log-row.KV_RESTORE[data-tag="KV MISS"] .log-tag { background: rgba(239,68,68,.15); color: var(--red); border-color: var(--red); }
.log-row.KV_RAM { border-left-color: var(--kv-save); }
.log-row.KV_RAM .log-tag { background: rgba(20,184,166,.15); color: var(--kv-save); border: 1px solid var(--kv-save); }
.log-row.KV_RAM[data-tag="KV EVICT RAM"] { border-left-color: var(--kv-evict); }
.log-row.KV_RAM[data-tag="KV EVICT RAM"] .log-tag { background: rgba(71,85,105,0.2); color: #94a3b8; border: 1px solid var(--kv-evict); }
.log-row.PREFILL { border-left-color: var(--prefill); }
.log-row.PREFILL .log-tag { background: rgba(234,179,8,.15); color: var(--prefill); border: 1px solid var(--prefill); }
.log-row.DECODE { border-left-color: var(--decode); }
.log-row.DECODE .log-tag { background: rgba(168,85,247,.15); color: var(--decode); border: 1px solid var(--decode); }
.log-row.DONE { border-left-color: #64748b; }
.log-row.DONE .log-tag { background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid #64748b; }
.log-row.CANCEL { border-left-color: var(--red); }
.log-row.CANCEL .log-tag { background: rgba(239,68,68,.2); color: var(--red); border: 1px solid var(--red); }
.table-wrap{flex:1;height:100%;min-height:0;overflow-y:auto}.hist-table{width:100%;border-collapse:collapse;font-size:11px;font-family:monospace}.hist-table th,.hist-table td{padding:5px 6px;text-align:left;border-bottom:1px solid var(--border);cursor:pointer}.hist-table th{color:var(--dim);position:sticky;top:0;background:var(--panel)}.hist-table tr:hover td{background:#131c2b}
.badge-eff-high{color:var(--reuse);font-weight:bold}.badge-eff-med{color:var(--prefill);font-weight:bold}.badge-eff-low{color:var(--red);font-weight:bold}
.badge-lost-zero{color:var(--dim);font-size:10px}.badge-lost-warn{background:rgba(239,68,68,.2);color:var(--red);padding:1px 5px;border-radius:4px;font-weight:bold}
.status-tag{padding:1px 5px;border-radius:4px;font-size:9.5px;font-weight:bold;display:inline-block}
.status-hot{background:rgba(34,197,94,.15);color:var(--reuse);border:1px solid rgba(34,197,94,.3)}
.status-rewind{background:rgba(249,115,22,.15);color:#f97316;border:1px solid rgba(249,115,22,.3)}
.status-ram{background:rgba(20,184,166,.15);color:var(--kv-save);border:1px solid rgba(20,184,166,.3)}
.status-cold{background:rgba(100,116,139,.15);color:var(--dim);border:1px solid var(--kv-evict)}
.skew-tag{color:#f59e0b;font-size:10px}
.hist-row-highlight { background: rgba(56,189,248,0.18) !important; outline: 1px solid var(--input); }
/* #4: status bar — level 1: connection to the server, level 2: the task phase */
.status-bar { display:flex; align-items:center; gap:10px; padding:6px 12px; background:var(--panel); border:1px solid var(--border); border-radius:9px; font-size:11px; flex-shrink:0; display:none !important; }
.status-dot { width:9px; height:9px; border-radius:50%; display:inline-block; background:var(--dim); flex-shrink:0; }
.dot-online { background:var(--reuse); box-shadow:0 0 6px var(--reuse); }
.dot-offline { background:var(--red); box-shadow:0 0 6px var(--red); }
.dot-playback { background:var(--prefill); box-shadow:0 0 6px var(--prefill); }
.tier-server-status { font-weight:bold; letter-spacing:.5px; color:var(--dim); }
.badge-mini-status { padding:2px 8px; border-radius:10px; font-size:10px; font-weight:bold; letter-spacing:.5px; }
.phase-IDLE { background:#1e293b; color:#94a3b8; }
.phase-INPUT { background:#0c4a6e; color:#7dd3fc; }
.phase-PREFILL { background:#713f12; color:#fde047; }
.phase-DECODE { background:#581c87; color:#d8b4fe; }
.phase-DONE { background:#14532d; color:#86efac; }
.queue-badge { margin-left:auto; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:bold; color:var(--prefill); background:rgba(234,179,8,.12); border:1px solid rgba(234,179,8,.4); }
/* #4: the player panel (PLAY / load-log); in LIVE the speed/"again" dim out */
.player-toolbar { display:flex; align-items:center; gap:8px; padding:6px 12px; background:var(--panel); border:1px solid var(--border); border-radius:9px; flex-shrink:0; display:none !important; }
.player-toolbar button { background:var(--panel2); border:1px solid var(--border); color:var(--text); font-family:inherit; font-size:12px; font-weight:700; padding:3px 10px; border-radius:5px; cursor:pointer; }
.player-toolbar button:disabled { opacity:.4; cursor:default; }
.player-toolbar select { background:var(--panel2); border:1px solid var(--border); color:var(--text); font-family:inherit; font-size:12px; padding:3px 6px; border-radius:5px; }
.player-toolbar select:disabled { opacity:.4; }
.ctl-file-label { font-size:11px; color:var(--dim); cursor:pointer; display:inline-flex; align-items:center; gap:4px; }
.ctl-file-label input[type=file] { font-size:11px; color:var(--text); max-width:160px; }
</style>
<!-- #14: the exact version (a floating @11 could have swapped the render behavior) -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.min.js"></script>
</head>
<body>
  <!-- #4: status bar — level 1: connection to the server, level 2: the task phase -->
  <div class="status-bar">
    <span id="tier-dot" class="status-dot dot-offline"></span>
    <span id="tier-server-status" class="tier-server-status">OFFLINE</span>
    <span id="status-badge" class="badge-mini-status badge-off">OFFLINE</span>
    <span id="queue-badge" class="queue-badge" style="display:none" data-i18n-title="queue_badge_title">QUEUE <span id="queue-val">0</span></span>
  </div>
  <div class="main-layout">
    <div class="left-col">
      <div class="card" style="padding:10px 12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-size:12px; font-weight:bold;">GPU VRAM KV-SLOT (<span id="hdr-ctx">—</span> CELLS)</span>
          <span style="font-size:11px; color:var(--dim);">Total Input: <b id="val-input" style="color:var(--input)">0</b> <span data-i18n="tok_unit">tok.</span></span>
        </div>
        <div class="donut-layout">
          <div class="donut-box">
            <canvas id="reactor-canvas" width="480" height="480"></canvas>
            <div class="ring-tooltip" id="ring-tooltip"></div>
            <div class="donut-core" id="donut-core-bg">
              <div id="core-phase" class="phase-badge">IDLE</div>
              <div id="core-speed" class="live-speed">—</div>
              <div id="core-pct" class="live-pct">—</div>
              <!-- M4: the hardware bottleneck of the phase (purely informational, no backend) -->
              <div id="core-bottleneck" data-i18n-title="bottleneck_title" style="font-size:10px; color:var(--dim); letter-spacing:.5px; margin-top:5px;">IDLE</div>
            </div>
          </div>
          <div class="donut-legend">
            <div class="legend-row" id="row-reuse">
              <div class="legend-title" style="color:var(--reuse)"><div class="dot" style="background:var(--reuse)"></div><span>KV REUSE:</span></div>
              <div class="legend-val" id="val-cached" style="color:var(--reuse)">0</div>
            </div>
            <div class="legend-row" id="row-prefill">
              <div class="legend-title" style="color:var(--prefill)"><div class="dot" style="background:var(--prefill)"></div><span>PREFILL:</span></div>
              <div class="legend-val" id="val-delta" style="color:var(--prefill)">0</div>
            </div>
            <div class="legend-row" id="row-decode">
              <div class="legend-title" style="color:var(--decode)"><div class="dot" style="background:var(--decode)"></div><span>DECODE:</span></div>
              <div class="legend-val" id="val-gen" style="color:var(--decode)">0</div>
            </div>
            <!-- M3: SAVED TIME — only with a MEASURED prefill speed (no magic constants) -->
            <div class="legend-row" id="row-kv-roi" style="background:transparent; border:none; padding:1px 0;">
              <div class="legend-title" style="color:var(--dim)"><span data-i18n-title="saved_time_title">SAVED TIME:</span></div>
              <div class="legend-val" id="lbl-kv-roi" style="color:var(--dim); font-size:11px">—</div>
            </div>
            <div class="legend-row" style="background:transparent; border:none; padding:1px 0;">
              <div class="legend-title" style="color:var(--dim)"><span>FREE:</span></div>
              <div class="legend-val" id="val-free" style="color:var(--dim)">—</div>
            </div>
            <!-- R6: legend of the outer rings (ring 2 — checkpoints, ring 3 — prompt cache) -->
            <div class="legend-row" style="background:transparent; border:none; padding:1px 0;">
              <div class="legend-title" style="color:var(--dim)"><span>CHECKPOINTS</span></div>
              <div class="legend-val" id="legend-cp" data-i18n-title="legend_cp_title" style="color:var(--dim);font-size:11px">—/—</div>
            </div>
            <div class="legend-row" style="background:transparent; border:none; padding:1px 0;">
              <div class="legend-title" style="color:var(--dim)"><span>PROMPT CACHE</span></div>
              <div class="legend-val" id="legend-ram" style="color:var(--dim);font-size:11px">N/A</div>
            </div>

            <!-- Separator -->
            <div style="border-top:1px solid rgba(255,255,255,0.08); margin:4px 0 2px 0;"></div>

            <!-- SPECULATIVE — SESSION CUMULATIVE: per-task SPEC TOTAL +
                 per-source CUMULATIVE (draft-mtp / ngram-mod, the counters
                 live for the whole server lifetime) -->
            <div class="legend-row" style="background:transparent; border:none; padding:1px 0;">
              <div class="legend-title" style="color:var(--dim); font-size:10px;"><span>SPEC TOTAL (task):</span></div>
              <div class="legend-val" id="lbl-spec-total" style="color:var(--dim); font-size:11px; font-weight:bold;">—</div>
            </div>
            <div class="legend-row" style="background:transparent; border:none; padding:1px 0;">
              <div class="legend-title" style="color:var(--dim); font-size:10px;"><span>DRAFT-MTP (session):</span></div>
              <div class="legend-val" id="lbl-spec-mtp" style="color:var(--dim); font-size:11px; font-weight:bold;">—</div>
            </div>
            <div class="legend-row" style="background:transparent; border:none; padding:1px 0;">
              <div class="legend-title" style="color:var(--dim); font-size:10px;"><span>NGRAM-MOD (session):</span></div>
              <div class="legend-val" id="lbl-ngram" style="color:var(--dim); font-size:11px; font-weight:bold;">—</div>
            </div>
            <div style="margin-top:2px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div id="lbl-mtp-pos" style="font-size:10.5px; font-weight:bold; color:var(--dim);">DRAFT-MTP ACC PER POS:</div>
                <span id="lbl-mtp-step" data-i18n="tok_per_step" style="font-size:10.5px; color:var(--dim); font-weight:bold;">— tok/step</span>
              </div>
              <div class="mtp-grid" id="mtp-bars" style="margin-top:3px;"></div>
            </div>
          </div>
        </div>
      </div>
      <div class="card card-history" style="padding:8px 12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <!-- Left: the title + the compact LEGEND button -->
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:11.5px; font-weight:bold;">REQUEST HISTORY</span>
            <span data-i18n="hist_limit" style="color:var(--dim); font-size:10px;">(up to 300)</span>
            <button id="open-metrics-help" class="btn-legend" data-i18n-title="legend_btn_title">📐 LEGEND</button>
            <button id="lang-toggle" class="btn-legend" data-i18n-title="lang_toggle_title">EN</button>
          </div>

          <a href="/api/logs" target="_blank" style="color:var(--input); font-size:10.5px; text-decoration:none;">📄 Raw log</a>
        </div>
        <div class="table-wrap">
          <table class="hist-table">
            <thead>
              <tr>
                <th>TIME</th><th>TASK</th>
                <th style="color:var(--input)" data-i18n-title="th_input_title">INPUT</th>
                <th style="color:var(--reuse)" data-i18n-title="th_kv_reuse_title">KV REUSE</th>
                <th data-i18n-title="th_eff_title">EFF%</th>
                <th data-i18n-title="th_prefix_gap_title">PREFIX GAP</th>
                <th data-i18n-title="th_status_title">STATUS</th>
                <th data-i18n-title="th_skew_title">SKEW</th>
                <th style="color:var(--prefill)">PREFILL</th>
                <th style="color:var(--decode)">DECODE</th>
                <th style="color:var(--mtp)" data-i18n-title="th_spec_title">SPEC</th>
              </tr>
            </thead>
            <tbody id="hist-live-row"></tbody>
            <tbody id="hist-rows"></tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="right-col">
      <div class="card card-log">
        <div class="card-log-collapsed-bar" id="log-collapsed-bar" data-i18n-title="log_expand_title">
          <span class="log-bar-label">EVENT LOG</span>
          <span class="log-bar-badge" id="log-bar-badge">0</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; position:relative;">
          <div style="font-size:12px; font-weight:bold;">EVENT LOG (GPU & RAM TELEMETRY)</div>
          <div style="display:flex; gap:6px; align-items:center;">
            <span data-i18n="log_depth" style="font-size:10.5px; color:var(--dim);">Depth: 500 events</span>
            <button id="btn-log-filter" style="background:#151e2b; border:1px solid var(--border); color:var(--text); border-radius:4px; font-size:11px; padding:2px 8px; cursor:pointer; display:flex; align-items:center; gap:4px;">
              <span data-i18n="filter_btn">🌪 Filter</span>
            </button>
            <button id="btn-toggle-log" style="background:#151e2b; border:1px solid var(--border); color:var(--dim); border-radius:4px; font-size:10px; padding:2px 6px; cursor:pointer;" data-i18n-title="log_collapse_title">▶</button>
          </div>

          <!-- The floating filter menu -->
          <div id="log-filter-popover" style="display:none; position:absolute; top:28px; right:0; width:220px; background:#0d1522; border:1px solid var(--border); box-shadow:0 8px 24px rgba(0,0,0,0.7); border-radius:6px; padding:10px; z-index:100; font-size:11px; font-family:-apple-system,BlinkMacSystemFont,monospace;">
            <div style="font-weight:bold; color:var(--text); margin-bottom:8px; border-bottom:1px solid var(--border); padding-bottom:4px; display:flex; justify-content:space-between;">
              <span data-i18n="filter_settings">Display settings</span>
              <span id="btn-filter-reset" data-i18n="filter_all" style="color:var(--input); cursor:pointer; font-weight:normal; font-size:10px;">All</span>
            </div>

            <!-- The time switch -->
            <label style="display:flex; align-items:center; gap:6px; margin-bottom:8px; cursor:pointer; color:var(--input);">
              <input type="checkbox" id="filter-show-time">
              <span data-i18n="filter_show_time">Show time [hh:mm:ss]</span>
            </label>

            <div data-i18n="filter_categories" style="color:var(--dim); font-size:10px; margin-bottom:4px; text-transform:uppercase;">Event categories:</div>
            <div style="display:flex; flex-direction:column; gap:4px;">
              <label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" class="tag-filter" data-target="INPUT" checked> <span style="color:var(--input)">INPUT</span> <span data-i18n="cat_input">(Input tokens)</span></label>
              <label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" class="tag-filter" data-target="PREFIX" checked> <span style="color:var(--prefix)">PREFIX</span> <span data-i18n="cat_prefix">(LCP match)</span></label>
              <label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" class="tag-filter" data-target="SLOT_SELECT" checked> <span style="color:var(--slot)">SLOT SELECT</span> <span data-i18n="cat_slot">(Slot selection)</span></label>
              <label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" class="tag-filter" data-target="KV_RESTORE" checked> <span style="color:var(--reuse)">KV REUSE</span> <span data-i18n="cat_kv">(Cache hits)</span></label>
              <label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" class="tag-filter" data-target="KV_RAM" checked> <span style="color:var(--kv-save)">KV RAM</span> <span data-i18n="cat_ram">(RAM checkpoints)</span></label>
              <label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" class="tag-filter" data-target="PREFILL" checked> <span style="color:var(--prefill)">PREFILL</span> <span data-i18n="cat_prefill">(Delta)</span></label>
              <label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" class="tag-filter" data-target="DECODE" checked> <span style="color:var(--decode)">DECODE</span> <span data-i18n="cat_decode">(Generation)</span></label>
              <label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" class="tag-filter" data-target="MTP" checked> <span style="color:var(--mtp)">MTP</span> <span data-i18n="cat_mtp">(Speculation)</span></label>
              <label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" class="tag-filter" data-target="DONE" checked> <span style="color:#94a3b8">DONE</span> <span data-i18n="cat_done">(Completion)</span></label>
            </div>
          </div>
        </div>
        <div class="log-box" id="log-feed"></div>
      </div>
    </div>
  </div>
  <!-- #4: the player panel (PLAY / load-log); in LIVE the speed/"again" are inactive -->
  <div class="player-toolbar">
    <button id="ctl-play" type="button" data-i18n-title="ctl_play_title">⏸</button>
    <button id="ctl-restart" type="button" data-i18n-title="ctl_restart_title">⏮</button>
    <button id="ctl-live" type="button" data-i18n-title="ctl_live_title" data-i18n="ctl_live" style="display:none">⏭ TO LIVE</button>
    <select id="ctl-speed" data-i18n-title="ctl_speed_title">
      <option value="1">1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
      <option value="8">8x</option>
      <option value="16">16x</option>
    </select>
    <button id="ctl-clear" type="button" data-i18n-title="ctl_clear_title" data-i18n="ctl_clear">Clear</button>
    <label class="ctl-file-label" data-i18n-title="ctl_file_title"><span data-i18n="ctl_file">Log</span><input id="ctl-file" type="file" accept=".log,.txt,.out"></label>
  </div>
  <!-- Offscreen source for the LEGEND modal (before mermaid.run()): invisible, but the browser computes the geometry -->
  <div style="position:fixed; left:-9999px; top:-9999px; width:1200px; visibility:hidden; pointer-events:none;">
    <pre class="mermaid flow-diagram">
flowchart LR
  A["INPUT<br/><small>Prompt · CPU</small>"]
  B["PREFIX / LCP<br/><small>Match search</small>"]
  C["KV REUSE / SOURCE<br/><small>Where is KV?</small>"]
  D["PREFILL<br/><small>New part · GPU</small>"]
  E["DECODE<br/><small>Generation · GEMV</small>"]
  F["SPEC (MTP+NGRAM)<br/><small>Draft ahead</small>"]
  G["DONE<br/><small>Response ready</small>"]
  H["HOT VRAM<br/><small>KV already in GPU</small>"]
  I["HOST RAM<br/><small>RAM → VRAM</small>"]
  J["COLD START<br/><small>No KV · full prefill</small>"]
  A --> B --> C
  C --> H --> D
  C --> I --> D
  C --> J --> D
  D --> E --> G
  E <--> F
  classDef in fill:#0d131d,stroke:#38bdf8,color:#f1f5f9,stroke-width:1.5px
  classDef pf fill:#0d131d,stroke:#6366f1,color:#f1f5f9,stroke-width:1.5px
  classDef kv fill:#0d131d,stroke:#22c55e,color:#f1f5f9,stroke-width:1.5px
  classDef pr fill:#0d131d,stroke:#eab308,color:#f1f5f9,stroke-width:1.5px
  classDef dc fill:#0d131d,stroke:#a855f7,color:#f1f5f9,stroke-width:1.5px
  classDef mt fill:#0d131d,stroke:#06b6d4,color:#f1f5f9,stroke-width:1.5px
  classDef dn fill:#0d131d,stroke:#64748b,color:#94a3b8,stroke-width:1px
  classDef hv fill:#0d131d,stroke:#22c55e,color:#94a3b8,stroke-width:1px,font-size:9px
  classDef hr fill:#0d131d,stroke:#14b8a6,color:#94a3b8,stroke-width:1px,font-size:9px
  classDef cs fill:#0d131d,stroke:#64748b,color:#94a3b8,stroke-width:1px,font-size:9px
  class A in
  class B pf
  class C kv
  class D pr
  class E dc
  class F mt
  class G dn
  class H hv
  class I hr
  class J cs
  linkStyle 0,1,8,9 stroke:#cbd5e1,stroke-width:1.5px
  linkStyle 2,3,4,5,6,7 stroke:#475569,stroke-width:1px
  linkStyle 10 stroke:#06b6d4,stroke-width:1px
    </pre>
  </div>
  <script>
  (() => {
    // I18N: en/ru. Default 'en'; persisted in localStorage 'wwwjs_lang'.
    // Static UI + mermaid diagram are translated by applyI18n(); events and
    // dynamic labels use t(key, vars) (server stores keys + params).
    const I18N = {
      en: {
        queue_badge_title: 'Deferred harness tasks (llamacpp:requests_deferred, /metrics)',
        tok_unit: 'tok.',
        bottleneck_title: 'Prefill = matrix×matrix (GEMM, TFLOPS bound); decode = matrix×vector (GEMV, memory-bandwidth bound GB/s)',
        saved_time_title: 'Time saved: restored tokens skipped prefill',
        legend_cp_title: 'M — checkpoint limit PER SLOT (--ctx-checkpoints)',
        tok_per_step: '— tok/step',
        hist_limit: '(up to 300)',
        legend_btn_title: 'Open the interactive inference diagram and metric formulas',
        lang_toggle_title: 'Switch language',
        th_input_title: 'Prompt size',
        th_kv_reuse_title: 'Taken from cache',
        th_eff_title: 'Cache efficiency (Hit Rate)',
        th_prefix_gap_title: 'Prefix break: tokens from the previous turn’s context that missed the cache (truncation)',
        th_status_title: 'Cache source and status',
        th_skew_title: 'Overpay due to checkpoint step (LCP − restored)',
        th_spec_title: 'SPEC TOTAL: accepted/generated (all speculative implementations)',
        log_expand_title: 'Expand EVENT LOG',
        log_depth: 'Depth: 500 events',
        filter_btn: '🌪 Filter',
        log_collapse_title: 'Collapse log',
        filter_settings: 'Display settings',
        filter_all: 'All',
        filter_show_time: 'Show time [hh:mm:ss]',
        filter_categories: 'Event categories:',
        cat_input: '(Input tokens)',
        cat_prefix: '(LCP match)',
        cat_slot: '(Slot selection)',
        cat_kv: '(Cache hits)',
        cat_ram: '(RAM checkpoints)',
        cat_prefill: '(Delta)',
        cat_decode: '(Generation)',
        cat_mtp: '(Speculation)',
        cat_done: '(Completion)',
        ctl_play_title: 'Pause / resume',
        ctl_restart_title: 'From the beginning',
        ctl_live_title: 'Switch to LIVE',
        ctl_live: '⏭ TO LIVE',
        ctl_speed_title: 'Playback speed',
        ctl_clear_title: 'Clear feed and history',
        ctl_clear: 'Clear',
        ctl_file_title: 'Load log (up to 2 MB)',
        ctl_file: 'Log',
        metrics_guide_btn: '📖 METRICS & FORMULAS',
        why_label: 'Why:',
        norm_label: 'Norm:',
        bottleneck_label: 'Bottleneck:',
        metric_input_title: 'Input context',
        metric_input_desc: 'Total volume of Token IDs received from the client (system prompt + message history + new question).',
        metric_input_code: 'prompt_tokens = task.n_tokens',
        metric_input_why: 'Baseline reference point. Shows how much data the model has to consume in total.',
        metric_kv_title: 'Cache hit',
        metric_kv_desc: 'Number of tokens whose keys/values (KV) already reside in VRAM or were restored from Host RAM checkpoints.',
        metric_kv_code: 'reused = cached_n_tokens || restored_checkpoint',
        metric_kv_why: 'The main accelerator. These tokens do not need to be recomputed on the GPU.',
        metric_eff_title: 'Cache Hit Rate',
        metric_eff_desc: 'Caching efficiency in percent. How "free" the request is for the GPU.',
        metric_eff_code: 'eff = (kv_reuse / input) * 100%',
        metric_eff_why: '≥98% — instant dialog response; <80% — cold start or topic change.',
        metric_gap_title: 'Prefix gap',
        metric_gap_desc: 'How many tokens from the previous turn’s context missed the current request’s cache.',
        metric_gap_code: 'gap = (prev_input + prev_decode) - reuse',
        metric_gap_why: 'Detects context truncation (agent frameworks, topic change). If > 0 — part of the prefix was lost.',
        metric_status_title: 'Physical KV source',
        metric_status_desc: 'Where the cache tensors physically come from: directly from a hot GPU slot, from RAM, or from scratch.',
        metric_status_code: 'HOT VRAM | RAM→VRAM (checkpoint) | COLD',
        metric_status_why_pre: 'Shows how the',
        metric_status_why_post: 'mechanism and the PCIe bus work during save/load.',
        metric_skew_title: 'Quantization-step overpay',
        metric_skew_desc: 'Difference between the exact prefix match (LCP) and the position of the nearest checkpoint in RAM.',
        metric_skew_code: 'skew = lcp_tokens - kv_restored_tokens',
        metric_skew_why: 'Shows how many tokens had to be recomputed on the GPU solely because of the checkpoint save step (spacing).',
        metric_prefill_title: 'New GPU compute',
        metric_prefill_desc: 'New tokens actually computed on the tensor cores. Parallel matrix multiplication (GEMM).',
        metric_prefill_code: 'delta = input - kv_reuse (ms / tokens = t/s)',
        metric_prefill_why: 'GPU TFLOPS limit. The smaller the delta — the faster generation starts.',
        metric_decode_title: 'Answer generation',
        metric_decode_desc: 'Step-by-step token output, one after another. Matrix-vector multiplication (GEMV).',
        metric_decode_code: 'tokens_generated / eval_time_seconds = t/s',
        metric_decode_why: 'GPU memory bus bandwidth (GB/s). Every token requires reading all model weights.',
        metric_spec_title: 'Speculative draft',
        metric_spec_desc: 'Multi-Token Prediction / n-gram. A small network or heuristic predicts several tokens ahead. SPEC TOTAL — all implementations combined (per-task); DRAFT-MTP / NGRAM-MOD — session CUMULATIVE.',
        metric_spec_code: 'acceptance_rate (accepted/generated) + mean_len = 1 + Σ acc_rate_per_pos',
        metric_spec_why: 'Allows obtaining several tokens per one weight-read tick from VRAM, speeding up DECODE.',
        // --- EVENT LOG: titles/details (server stores keys, client renders via t()) ---
        ev_nctx_title: 'n_ctx mismatch',
        ev_nctx_detail: 'Sources diverged: <b>{a}</b> vs <b>{b}</b> — parser bug?',
        ev_lcp_title: 'LCP history match',
        ev_lcp_detail: 'Match found: <b>{pct}%</b>',
        ev_lcp_detail_tok: 'Match found: <b>{pct}%</b> ({tok} tok.)',
        ev_slot_lcp_title: 'Slot selected by LCP',
        ev_slot_lcp_detail: 'LCP <b>{lcp}</b> / {prompt} tok. ({pct}%)',
        ev_slot_lcp_detail_slot: 'Slot <b>#{slot}</b> · LCP <b>{lcp}</b> / {prompt} tok. ({pct}%)',
        ev_slot_lru_title: 'Slot selected (LRU)',
        ev_slot_lru_detail: 'Slot <b>#{slot}</b> · no similar prefix (Cold Start)',
        ev_prefill_done_title: 'Prefill complete',
        ev_prefill_done_est: 'Computed on GPU: ~<b>{tok}</b> tok. (LCP estimate), generation starts.',
        ev_prefill_done_detail: 'Total: <b>{total}</b> | KV reused: <b>{reused}</b> | NEW: <b>{new}</b> tok. in {ms} ms (<b>{tps} tok/s</b>), generation starts.',
        ev_prefill_done_measured: 'Total: <b>{total}</b> | KV reused: <b>{reused}</b> | NEW: <b>{new}</b> tok. in {ms} ms (<b>{tps} tok/s</b>) (measured)',
        ev_decode_start_title: 'Generation started',
        ev_decode_start_detail: 'Output started (speed: <b>{tps} t/s</b>)...',
        ev_decode_instant_detail: 'Output finished instantly (speed: <b>{tps} t/s</b>)...',
        ev_input_title: 'Request #{tid}',
        ev_input_detail: 'Received <b>{tok}</b> Token IDs on CPU',
        ev_kv_restore_title: 'Cache restored from RAM',
        ev_kv_restore_detail: 'Restored <b>{tok}</b> tok. ({size} MiB) from <b>Host RAM</b> to VRAM.',
        ev_full_reproc_title: 'LCP match, but full recompute',
        ev_full_reproc_detail: 'VRAM: <b>0</b> tok. restored (LCP <b>{lcp}</b>). Full prefill <b>{total}</b> tok.',
        ev_kv_miss_title: 'Cache miss / Cold start',
        ev_kv_miss_detail: 'VRAM: <b>0</b> tok., RAM checkpoint: not found. Full prefill <b>{total}</b> tok.',
        ev_cold_start_title: 'Cold start',
        ev_cold_start_detail: 'VRAM: <b>0</b> tok., RAM checkpoint: not found. Full prefill from scratch.',
        ev_cached_prefix_title: 'Cached prefix',
        ev_cached_prefix_detail: 'Slot context already has <b>{tok}</b> tok. — their prefill is skipped (source: {source}).',
        ev_cp_save_title: 'Checkpoint saved to RAM',
        ev_cp_save_detail: 'Slot <b>{n} of {m}</b>: <b>{tok}</b> tok. (<b>{size} MiB</b>)',
        ev_cp_loop_title: '5 identical erases in a row',
        ev_cp_loop_detail: 'Task <b>{tid}</b>: checkpoint <b>pos {posMin}–{posMax}</b> erased <b>5</b> times in a row (identical erase lines).',
        ev_cp_loop_detail_stall: 'Task <b>{tid}</b>: checkpoint <b>pos {posMin}–{posMax}</b> erased <b>5</b> times in a row (identical erase lines). Slot is is_processing=true now — possible stall.',
        ev_cp_evict_title: 'Checkpoint evicted from RAM',
        ev_cp_evict_detail: '<b>{tok}</b> tok. (<b>{size} MiB</b>) — reason: <b>{reason}</b>',
        ev_cp_supersede_title: 'Checkpoint superseded',
        ev_cp_supersede_detail: 'Checkpoint at <b>{tok} tok.</b> replaced by a new one at the same point',
        ev_prompt_save_title: 'Dialog snapshot',
        ev_prompt_save_detail: '<b>{tok}</b> tok. pinned in Host RAM ({size} MiB)',
        ev_prefill_prog_title: 'Prefill {pct}%',
        ev_prefill_prog_detail: 'Processed <b>{tok}</b> tok.',
        ev_prefill_prog_detail_spd: 'Processed <b>{tok}</b> tok. | Speed: <b>{tps} tok/s</b>',
        ev_mtp_title: 'Speculative step (SPEC TOTAL)',
        ev_mtp_detail: 'Draft: <b>{gen}</b> generated, <b>{acc}</b> accepted (<b>{pct}%</b>)',
        ev_mtp_detail_len: 'Draft: <b>{gen}</b> generated, <b>{acc}</b> accepted (<b>{pct}%</b>) | Mean len: <b>{len}</b> tok/step',
        ev_cancel_title: 'Request cancelled',
        ev_cancel_detail: 'Generation interrupted by user (task <b>{tid}</b>).',
        ev_ctx_shift_title: 'Context shift',
        ev_ctx_shift_detail: 'truncated=1 after context shift (n_keep/n_discard) — not an overflow',
        ev_overflow_title: 'Context limit',
        ev_overflow_detail: 'Context overflow!',
        ev_done_title: 'Response complete',
        ev_done_detail: 'Generated <b>{tok}</b> tok. in {sec} s | avg. <b>{tps} t/s</b>',
        ev_done_detail_peak: 'Generated <b>{tok}</b> tok. in {sec} s | avg. <b>{tps} t/s</b> | peak <b>{peak} t/s</b>',
        ev_stall_title: 'Long prefill pause',
        ev_stall_detail: 'Slot busy for <b>{sec}s</b> without moving to detection/decode!',
        src_checkpoint: 'checkpoint (Host RAM → VRAM)',
        src_full_reprocess: 'none (full re-processing)',
        src_cold_start: 'none (cold start)',
        src_prompt_cache: 'prompt-cache (Host RAM)',
        src_hot_slot: 'hot slot (KV in VRAM)',
        src_unknown: 'unknown',
        reason_invalidated: 'invalidated (context invalid)',
        reason_capacity: 'capacity (oldest evicted on fill)',
        reason_too_close: 'capacity (too close to neighbor)',
        tok: 'tok',
        tok_step: 'tok/step',
        tok_wait: 'tok (awaiting measurement)',
        sys_events: 'Server system events',
        tt_cp_created: 'created',
        tt_cp_restored: 'restored ×',
        tt_cp_evicted: 'evicted',
        tt_cp_partial: 'partial state, size does not depend on position',
        tt_cp_count: 'checkpoints',
        tt_cp_restored_vram: 'context checkpoint restored (Host RAM → VRAM)',
        tt_lost_expected: 'Expected in cache: {n} tok.',
        st_reproc: 'LCP matched, but the server recomputed the whole prompt (KV not restored)',
        st_hotcp: 'Rewind to checkpoint inside the VRAM slot',
        st_ramvram: 'Restored from Host RAM',
        st_hot: 'Direct VRAM hit',
        st_ram: 'Loaded from prompt cache (Host RAM)',
        st_cache: 'Source unknown',
        st_skew: 'LCP matched at {lcp}, restored {reuse} — the difference was recomputed in prefill',
        desync_title: 'DESYNC at token #{pos}',
        desync_slot: 'Slot had',
        desync_req: 'Request sent',
        alert_log_big: 'Log too large: {mb} MB (limit 2 MB)',
        alert_load_fail: 'Failed to load log',
        alert_load_fail_http: 'Failed to load log: HTTP {status}',
        mermaid_unavailable: 'Diagram unavailable (Mermaid not loaded)',
      },
      ru: {
        queue_badge_title: 'Отложенные задачи harness (llamacpp:requests_deferred, /metrics)',
        tok_unit: 'ток.',
        bottleneck_title: 'Prefill = матрица×матрица (GEMM, лимит TFLOPS); decode = матрица×вектор (GEMV, лимит шины памяти GB/s)',
        saved_time_title: 'Экономия времени: восстановленные токены не прошли prefill',
        legend_cp_title: 'M — лимит чекпоинтов НА СЛОТ (--ctx-checkpoints)',
        tok_per_step: '— ток/такт',
        hist_limit: '(до 300)',
        legend_btn_title: 'Открыть интерактивную схему инференса и формулы метрик',
        lang_toggle_title: 'Сменить язык',
        th_input_title: 'Размер промпта',
        th_kv_reuse_title: 'Взято из кэша',
        th_eff_title: 'Эффективность кэша (Hit Rate)',
        th_prefix_gap_title: 'Разрыв префикса: токены из контекста прошлого хода, не попавшие в кэш (обрезка)',
        th_status_title: 'Источник и статус кэша',
        th_skew_title: 'Переплата из-за шага чекпоинта (LCP − восстановлено)',
        th_spec_title: 'SPEC TOTAL: принято/сгенерировано (все speculative-реализации)',
        log_expand_title: 'Развернуть EVENT LOG',
        log_depth: 'Глубина: 500 событий',
        filter_btn: '🌪 Фильтр',
        log_collapse_title: 'Свернуть лог',
        filter_settings: 'Настройки отображения',
        filter_all: 'Все',
        filter_show_time: 'Показывать время [hh:mm:ss]',
        filter_categories: 'Категории событий:',
        cat_input: '(Входные токены)',
        cat_prefix: '(Сверка LCP)',
        cat_slot: '(Выбор слота)',
        cat_kv: '(Хиты кэша)',
        cat_ram: '(Чекпоинты RAM)',
        cat_prefill: '(Дельта)',
        cat_decode: '(Генерация)',
        cat_mtp: '(Спекуляция)',
        cat_done: '(Завершение)',
        ctl_play_title: 'Пауза / продолжить',
        ctl_restart_title: 'Сначала',
        ctl_live_title: 'Переключиться в LIVE',
        ctl_live: '⏭ В LIVE',
        ctl_speed_title: 'Скорость воспроизведения',
        ctl_clear_title: 'Очистить ленту и историю',
        ctl_clear: 'Очистить',
        ctl_file_title: 'Загрузить лог (до 2 МБ)',
        ctl_file: 'Лог',
        metrics_guide_btn: '📖 МЕТРИКИ И ФОРМУЛЫ',
        why_label: 'Зачем:',
        norm_label: 'Норма:',
        bottleneck_label: 'Bottleneck:',
        metric_input_title: 'Входной контекст',
        metric_input_desc: 'Полный объём Token IDs, пришедших с клиента (системный промпт + история сообщений + новый вопрос).',
        metric_input_code: 'prompt_tokens = task.n_tokens',
        metric_input_why: 'Базовая точка отсчета. Показывает, сколько всего данных нужно скормить модели.',
        metric_kv_title: 'Попадание в кэш',
        metric_kv_desc: 'Число токенов, ключи/значения (KV) которых уже лежат в VRAM или подняты из Host RAM чекпоинтов.',
        metric_kv_code: 'reused = cached_n_tokens || restored_checkpoint',
        metric_kv_why: 'Главный ускоритель. Эти токены не требуют пересчета на GPU.',
        metric_eff_title: 'Cache Hit Rate',
        metric_eff_desc: 'Эффективность кэширования в процентах. Насколько запрос «бесплатен» для видеокарты.',
        metric_eff_code: 'eff = (kv_reuse / input) * 100%',
        metric_eff_why: '≥98% — мгновенный ответ диалога; <80% — холодный старт или смена темы.',
        metric_gap_title: 'Разрыв префикса',
        metric_gap_desc: 'Сколько токенов из контекста предыдущего хода не попало в кэш текущего запроса.',
        metric_gap_code: 'gap = (prev_input + prev_decode) - reuse',
        metric_gap_why: 'детектирует обрезку контекста (агентские фреймворки, смена темы). Если > 0 — часть префикса потеряна.',
        metric_status_title: 'Физический источник KV',
        metric_status_desc: 'Откуда физически взяты тензоры кэша: прямо из горячего слота GPU, из RAM или с нуля.',
        metric_status_code: 'HOT VRAM | RAM→VRAM (чекпоинт) | COLD',
        metric_status_why_pre: 'Показывает работу механизма',
        metric_status_why_post: 'и шины PCIe при выгрузке/загрузке.',
        metric_skew_title: 'Переплата за квантование шага',
        metric_skew_desc: 'Разница между точным совпадением префикса (LCP) и позицией ближайшего чекпоинта в RAM.',
        metric_skew_code: 'skew = lcp_tokens - kv_restored_tokens',
        metric_skew_why: 'Показывает, сколько токенов пришлось досчитать на GPU только из-за шага сохранения чекпоинта (spacing).',
        metric_prefill_title: 'Новый расчет GPU',
        metric_prefill_desc: 'Реально вычисленные на тензорных ядрах новые токены. Параллельное умножение матриц (GEMM).',
        metric_prefill_code: 'delta = input - kv_reuse (ms / tokens = t/s)',
        metric_prefill_why: 'Лимит TFLOPS видеокарты. Чем меньше дельта — тем быстрее старт генерации.',
        metric_decode_title: 'Генерация ответа',
        metric_decode_desc: 'Пошаговый вывод токенов один за другим. Умножение матрицы на вектор (GEMV).',
        metric_decode_code: 'tokens_generated / eval_time_seconds = t/s',
        metric_decode_why: 'Пропускная способность шины памяти GPU (GB/s). Каждому токену нужно прочитать все веса модели.',
        metric_spec_title: 'Спекулятивный драфт',
        metric_spec_desc: 'Multi-Token Prediction / n-gram. Маленькая сеть или эвристика предугадывает несколько токенов вперёд. SPEC TOTAL — все реализации вместе (per-task); DRAFT-MTP / NGRAM-MOD — CUMULATIVE по сессии.',
        metric_spec_code: 'acceptance_rate (принято/сгенерировано) + mean_len = 1 + Σ acc_rate_per_pos',
        metric_spec_why: 'Позволяет получать несколько токенов за 1 такт чтения весов из VRAM, ускоряя DECODE.',
        // --- EVENT LOG: заголовки/детали (сервер хранит ключи, клиент рендерит через t()) ---
        ev_nctx_title: 'n_ctx mismatch',
        ev_nctx_detail: 'Источники разошлись: <b>{a}</b> vs <b>{b}</b> — баг парсера?',
        ev_lcp_title: 'LCP Сверка истории',
        ev_lcp_detail: 'Найдено совпадение: <b>{pct}%</b>',
        ev_lcp_detail_tok: 'Найдено совпадение: <b>{pct}%</b> ({tok} ток.)',
        ev_slot_lcp_title: 'Выбор слота по LCP',
        ev_slot_lcp_detail: 'LCP <b>{lcp}</b> / {prompt} ток. ({pct}%)',
        ev_slot_lcp_detail_slot: 'Слот <b>#{slot}</b> · LCP <b>{lcp}</b> / {prompt} ток. ({pct}%)',
        ev_slot_lru_title: 'Выбор слота (LRU)',
        ev_slot_lru_detail: 'Слот <b>#{slot}</b> · похожего префикса нет (Cold Start)',
        ev_prefill_done_title: 'Префилл завершен',
        ev_prefill_done_est: 'Посчитано на GPU: ~<b>{tok}</b> ток. (оценка по LCP), начинается генерация.',
        ev_prefill_done_detail: 'Total: <b>{total}</b> | KV reused: <b>{reused}</b> | NEW: <b>{new}</b> ток. за {ms} мс (<b>{tps} tok/s</b>), начинается генерация.',
        ev_prefill_done_measured: 'Total: <b>{total}</b> | KV reused: <b>{reused}</b> | NEW: <b>{new}</b> ток. за {ms} мс (<b>{tps} tok/s</b>) (измерено)',
        ev_decode_start_title: 'Старт генерации',
        ev_decode_start_detail: 'Вывод начался (скорость: <b>{tps} t/s</b>)...',
        ev_decode_instant_detail: 'Вывод завершен мгновенно (скорость: <b>{tps} t/s</b>)...',
        ev_input_title: 'Запрос #{tid}',
        ev_input_detail: 'Получено <b>{tok}</b> Token IDs на CPU',
        ev_kv_restore_title: 'Кэш поднят из RAM',
        ev_kv_restore_detail: 'Восстановлено <b>{tok}</b> ток. ({size} MiB) из <b>Host RAM</b> в VRAM.',
        ev_full_reproc_title: 'LCP-совпадение, но полный пересчёт',
        ev_full_reproc_detail: 'VRAM: <b>0</b> ток. восстановлено (LCP <b>{lcp}</b>). Полный prefill <b>{total}</b> ток.',
        ev_kv_miss_title: 'Промах кэша / Cold start',
        ev_kv_miss_detail: 'VRAM: <b>0</b> ток., RAM-чекпоинт: не найден. Полный prefill <b>{total}</b> ток.',
        ev_cold_start_title: 'Cold start',
        ev_cold_start_detail: 'VRAM: <b>0</b> ток., RAM-чекпоинт: не найден. Полный prefill с нуля.',
        ev_cached_prefix_title: 'Кэшированный префикс',
        ev_cached_prefix_detail: 'В контексте слота уже <b>{tok}</b> ток. — их prefill не выполняется (источник: {source}).',
        ev_cp_save_title: 'Чекпоинт сохранен в RAM',
        ev_cp_save_detail: 'Слот <b>{n} of {m}</b>: <b>{tok}</b> ток. (<b>{size} MiB</b>)',
        ev_cp_loop_title: '5 одинаковых erase подряд',
        ev_cp_loop_detail: 'Таск <b>{tid}</b>: чекпоинт <b>pos {posMin}–{posMax}</b> стёрт <b>5</b> раз подряд (одинаковые erase-строки).',
        ev_cp_loop_detail_stall: 'Таск <b>{tid}</b>: чекпоинт <b>pos {posMin}–{posMax}</b> стёрт <b>5</b> раз подряд (одинаковые erase-строки). Слот сейчас is_processing=true — возможен stall.',
        ev_cp_evict_title: 'Чекпоинт вытеснен из RAM',
        ev_cp_evict_detail: '<b>{tok}</b> ток. (<b>{size} MiB</b>) — причина: <b>{reason}</b>',
        ev_cp_supersede_title: 'Чекпоинт заменён',
        ev_cp_supersede_detail: 'Чекпоинт на <b>{tok} ток.</b> заменён новым на той же точке',
        ev_prompt_save_title: 'Снимок диалога',
        ev_prompt_save_detail: '<b>{tok}</b> ток. зафиксировано в Host RAM ({size} MiB)',
        ev_prefill_prog_title: 'Префилл {pct}%',
        ev_prefill_prog_detail: 'Обработано <b>{tok}</b> ток.',
        ev_prefill_prog_detail_spd: 'Обработано <b>{tok}</b> ток. | Скорость: <b>{tps} tok/s</b>',
        ev_mtp_title: 'Спекулятивный шаг (SPEC TOTAL)',
        ev_mtp_detail: 'Draft: <b>{gen}</b> сгенерировано, <b>{acc}</b> принято (<b>{pct}%</b>)',
        ev_mtp_detail_len: 'Draft: <b>{gen}</b> сгенерировано, <b>{acc}</b> принято (<b>{pct}%</b>) | Длина: <b>{len}</b> ток/такт',
        ev_cancel_title: 'Запрос отменен',
        ev_cancel_detail: 'Генерация прервана пользователем (task <b>{tid}</b>).',
        ev_ctx_shift_title: 'Сдвиг контекста',
        ev_ctx_shift_detail: 'truncated=1 после context shift (n_keep/n_discard) — не переполнение',
        ev_overflow_title: 'Лимит контекста',
        ev_overflow_detail: 'Контекст переполнен!',
        ev_done_title: 'Ответ завершен',
        ev_done_detail: 'Сгенерировано <b>{tok}</b> ток. за {sec} с | средн. <b>{tps} t/s</b>',
        ev_done_detail_peak: 'Сгенерировано <b>{tok}</b> ток. за {sec} с | средн. <b>{tps} t/s</b> | пик <b>{peak} t/s</b>',
        ev_stall_title: 'Долгая пауза префилла',
        ev_stall_detail: 'Слот занят уже <b>{sec}с</b> без перехода к детекции/декоду!',
        src_checkpoint: 'checkpoint (Host RAM → VRAM)',
        src_full_reprocess: 'нет (full re-processing)',
        src_cold_start: 'нет (cold start)',
        src_prompt_cache: 'prompt-cache (Host RAM)',
        src_hot_slot: 'горячий слот (KV в VRAM)',
        src_unknown: 'не определён',
        reason_invalidated: 'invalidated (контекст невалиден)',
        reason_capacity: 'capacity (вытеснен старейший при заполнении)',
        reason_too_close: 'capacity (too close к соседнему)',
        tok: 'ток',
        tok_step: 'ток/такт',
        tok_wait: 'ток (ожидание замера)',
        sys_events: 'Системные события сервера',
        tt_cp_created: 'создан',
        tt_cp_restored: 'восстановлен ×',
        tt_cp_evicted: 'вытеснен',
        tt_cp_partial: 'частичное состояние, размер не зависит от позиции',
        tt_cp_count: 'чекпоинтов',
        tt_cp_restored_vram: 'context checkpoint восстановлен (Host RAM → VRAM)',
        tt_lost_expected: 'Ожидалось в кэше: {n} ток.',
        st_reproc: 'LCP совпал, но сервер пересчитал весь промпт (KV не восстановлен)',
        st_hotcp: 'Откат внутри VRAM-слота к чекпоинту',
        st_ramvram: 'Восстановлено из Host RAM',
        st_hot: 'Прямое попадание в VRAM',
        st_ram: 'Загружено из prompt cache (Host RAM)',
        st_cache: 'Источник не определён',
        st_skew: 'LCP совпал на {lcp}, восстановлено {reuse} — разница пересчитана в prefill',
        desync_title: 'РАССИНХРОН на токене #{pos}',
        desync_slot: 'В слоте было',
        desync_req: 'В запросе пришло',
        alert_log_big: 'Лог слишком большой: {mb} МБ (лимит 2 МБ)',
        alert_load_fail: 'Не удалось загрузить лог',
        alert_load_fail_http: 'Не удалось загрузить лог: HTTP {status}',
        mermaid_unavailable: 'Диаграмма недоступна (Mermaid не загружен)',
      },
    };
    // Mermaid diagram source per language (no backticks — inside HTML_PAGE template literal)
    const I18N_MERMAID = {
      en: ['flowchart LR',
        '  A["INPUT<br/><small>Prompt · CPU</small>"]',
        '  B["PREFIX / LCP<br/><small>Match search</small>"]',
        '  C["KV REUSE / SOURCE<br/><small>Where is KV?</small>"]',
        '  D["PREFILL<br/><small>New part · GPU</small>"]',
        '  E["DECODE<br/><small>Generation · GEMV</small>"]',
        '  F["SPEC (MTP+NGRAM)<br/><small>Draft ahead</small>"]',
        '  G["DONE<br/><small>Response ready</small>"]',
        '  H["HOT VRAM<br/><small>KV already in GPU</small>"]',
        '  I["HOST RAM<br/><small>RAM → VRAM</small>"]',
        '  J["COLD START<br/><small>No KV · full prefill</small>"]',
        '  A --> B --> C',
        '  C --> H --> D',
        '  C --> I --> D',
        '  C --> J --> D',
        '  D --> E --> G',
        '  E <--> F',
        '  classDef in fill:#0d131d,stroke:#38bdf8,color:#f1f5f9,stroke-width:1.5px',
        '  classDef pf fill:#0d131d,stroke:#6366f1,color:#f1f5f9,stroke-width:1.5px',
        '  classDef kv fill:#0d131d,stroke:#22c55e,color:#f1f5f9,stroke-width:1.5px',
        '  classDef pr fill:#0d131d,stroke:#eab308,color:#f1f5f9,stroke-width:1.5px',
        '  classDef dc fill:#0d131d,stroke:#a855f7,color:#f1f5f9,stroke-width:1.5px',
        '  classDef mt fill:#0d131d,stroke:#06b6d4,color:#f1f5f9,stroke-width:1.5px',
        '  classDef dn fill:#0d131d,stroke:#64748b,color:#94a3b8,stroke-width:1px',
        '  classDef hv fill:#0d131d,stroke:#22c55e,color:#94a3b8,stroke-width:1px,font-size:9px',
        '  classDef hr fill:#0d131d,stroke:#14b8a6,color:#94a3b8,stroke-width:1px,font-size:9px',
        '  classDef cs fill:#0d131d,stroke:#64748b,color:#94a3b8,stroke-width:1px,font-size:9px',
        '  class A in',
        '  class B pf',
        '  class C kv',
        '  class D pr',
        '  class E dc',
        '  class F mt',
        '  class G dn',
        '  class H hv',
        '  class I hr',
        '  class J cs',
        '  linkStyle 0,1,8,9 stroke:#cbd5e1,stroke-width:1.5px',
        '  linkStyle 2,3,4,5,6,7 stroke:#475569,stroke-width:1px',
        '  linkStyle 10 stroke:#06b6d4,stroke-width:1px',
      ].join('\\n'),
      ru: ['flowchart LR',
        '  A["INPUT<br/><small>Промпт · CPU</small>"]',
        '  B["PREFIX / LCP<br/><small>Поиск совпадения</small>"]',
        '  C["KV REUSE / SOURCE<br/><small>Where is KV?</small>"]',
        '  D["PREFILL<br/><small>Новая часть · GPU</small>"]',
        '  E["DECODE<br/><small>Генерация · GEMV</small>"]',
        '  F["SPEC (MTP+NGRAM)<br/><small>Драфт вперёд</small>"]',
        '  G["DONE<br/><small>Ответ готов</small>"]',
        '  H["HOT VRAM<br/><small>KV уже в GPU</small>"]',
        '  I["HOST RAM<br/><small>RAM → VRAM</small>"]',
        '  J["COLD START<br/><small>KV нет · полный prefill</small>"]',
        '  A --> B --> C',
        '  C --> H --> D',
        '  C --> I --> D',
        '  C --> J --> D',
        '  D --> E --> G',
        '  E <--> F',
        '  classDef in fill:#0d131d,stroke:#38bdf8,color:#f1f5f9,stroke-width:1.5px',
        '  classDef pf fill:#0d131d,stroke:#6366f1,color:#f1f5f9,stroke-width:1.5px',
        '  classDef kv fill:#0d131d,stroke:#22c55e,color:#f1f5f9,stroke-width:1.5px',
        '  classDef pr fill:#0d131d,stroke:#eab308,color:#f1f5f9,stroke-width:1.5px',
        '  classDef dc fill:#0d131d,stroke:#a855f7,color:#f1f5f9,stroke-width:1.5px',
        '  classDef mt fill:#0d131d,stroke:#06b6d4,color:#f1f5f9,stroke-width:1.5px',
        '  classDef dn fill:#0d131d,stroke:#64748b,color:#94a3b8,stroke-width:1px',
        '  classDef hv fill:#0d131d,stroke:#22c55e,color:#94a3b8,stroke-width:1px,font-size:9px',
        '  classDef hr fill:#0d131d,stroke:#14b8a6,color:#94a3b8,stroke-width:1px,font-size:9px',
        '  classDef cs fill:#0d131d,stroke:#64748b,color:#94a3b8,stroke-width:1px,font-size:9px',
        '  class A in',
        '  class B pf',
        '  class C kv',
        '  class D pr',
        '  class E dc',
        '  class F mt',
        '  class G dn',
        '  class H hv',
        '  class I hr',
        '  class J cs',
        '  linkStyle 0,1,8,9 stroke:#cbd5e1,stroke-width:1.5px',
        '  linkStyle 2,3,4,5,6,7 stroke:#475569,stroke-width:1px',
        '  linkStyle 10 stroke:#06b6d4,stroke-width:1px',
      ].join('\\n'),
    };
    let LANG = 'en';
    try {
      const savedLang = localStorage.getItem('wwwjs_lang');
      if (savedLang === 'ru' || savedLang === 'en') LANG = savedLang;
    } catch (e) { /* no localStorage — default en */ }
    function t(key, vars) {
      let s = I18N[LANG][key];
      if (s == null) s = I18N.en[key];
      if (s == null) s = key;
      if (vars) for (const k of Object.keys(vars)) s = s.split('{' + k + '}').join(vars[k]);
      return s;
    }
    function applyI18n() {
      const qsa = document.querySelectorAll ? document.querySelectorAll.bind(document) : () => [];
      qsa('[data-i18n]').forEach(el => { el.textContent = t(el.getAttribute('data-i18n')); });
      qsa('[data-i18n-title]').forEach(el => { el.title = t(el.getAttribute('data-i18n-title')); });
      const pre = document.querySelector ? document.querySelector('.mermaid.flow-diagram') : null;
      if (pre) {
        pre.textContent = I18N_MERMAID[LANG];
        if (window.mermaid) mermaid.run().catch((e) => console.error('mermaid render:', e));
      }
      const lt = document.getElementById('lang-toggle');
      if (lt) lt.textContent = LANG.toUpperCase();
    }
    window.__wwwT = t; // shared with the LEGEND overlay script (fallback text)
    const langToggle = document.getElementById('lang-toggle');
    if (langToggle) langToggle.addEventListener('click', () => {
      LANG = LANG === 'en' ? 'ru' : 'en';
      try { localStorage.setItem('wwwjs_lang', LANG); } catch (e) {}
      applyI18n();
    });

    // Educational diagram: static Mermaid (no runtime data), CDN v11
    if (window.mermaid) {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'base',
        themeVariables: {
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace',
          fontSize: '10px',
          lineColor: '#475569',
        },
        flowchart: {
          htmlLabels: true,
          curve: 'linear',
          nodeSpacing: 22,
          rankSpacing: 38,
          padding: 10,
        },
      });
    }
    applyI18n(); // initial render: static UI + mermaid diagram in the selected language
    const buildRingModel = ${buildRingModel.toString()};
    const FX = ${JSON.stringify(FX)};
    const clamp01 = ${clamp01.toString()};
    const decay = ${decay.toString()};
    const fx = ${fx.toString()};
    const computeRing3Layout = ${computeRing3Layout.toString()};
    // ?fx=off or prefers-reduced-motion -> the effects are static (no flashes)
    const fxOff = new URLSearchParams(location.search).get('fx') === 'off';
    const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const fxEnabled = !fxOff && !reducedMotion;
    const cvs = document.getElementById('reactor-canvas');
    const ctx = cvs.getContext('2d');
    const CX = 240, CY = 240;
    // Hierarchy (outside in): VRAM → checkpoints → Host RAM.
    // Golden layout (φ≈1.618): thicknesses 10→16→26, gaps 8→13 (Fibonacci).
    // The mass grows from the center to the edge: RAM (sparse) → checkpoints → VRAM (dense).
    const R = 201, LW = 26;   // outer: the GPU VRAM KV slot (188–214px)
    const R2 = 167, LW2 = 16; // middle: checkpoints (159–175px)
    const R3 = 146, LW3 = 10; // inner: the Host RAM prompt cache (141–151px)
    const START = -Math.PI / 2;
    // Geometric invariant: an arc shorter than 2 px at LW=26 is drawn as a
    // radial "needle" (the stroke is thicker than the path length) — we do not pass it to Canvas
    const MIN_ARC_PX = 2;
    // Vortex: a clockwise tangential shift of the injector beam's end (rad).
    // The beam does not merely connect the rings, it "unwinds" the outer VRAM orbit.
    // 0 → a radial spoke (the old behavior).
    const BEAM_SWEEP = 0.16;

    let anim = { cached: 0, delta: 0, gen: 0, total: 0 };
    let target = { cached: 0, delta: 0, gen: 0, total: 0 };
    let animRunning = false;
    function kick() {
      if (animRunning) return;
      animRunning = true;
      requestAnimationFrame(frame);
    }

    // The static model of the outer rings (R3). Rebuilt in renderState,
    // kick only on a model change (the P5 gate: in idle the frame does not spin).
    let ringModel = { cp: { marks: [], count: 0, max: null, state: 'na' },
                      ram: { items: [], k: 1, used: null, limit: null, tokensUsed: 0, limitTokens: null, limitTokensEff: null, state: 'na' },
                      until: 0 };
    let lastRingModelHash = '';
    // The last drawn ring-3 layout (for hover, B2)
    let ring3Layout = [];

    const mtpContainer = document.getElementById('mtp-bars');
    let renderedMtpN = 0;
    function buildMtpBars(n) {
      mtpContainer.innerHTML = Array.from({length: n}, (_, i) =>
        '<div class="mtp-item">' +
          '<div class="mtp-col"><div class="mtp-fill" id="mtp-f' + i + '" style="height:0%"></div></div>' +
          '<div style="margin-top:2px; color:var(--dim); font-size:9.5px;">P' + (i + 1) + '</div>' +
          '<div id="mtp-v' + i + '" style="color:var(--mtp-dim);font-weight:bold">—</div>' +
        '</div>'
      ).join('');
      renderedMtpN = n;
    }
    // #1: the scale is built in renderState from specConfig.draftNMax (the startup log)
    // + the observed specMtp.posAcc — without data and without configuration there are no bars
    buildMtpBars(0);

    // Smoothing of the inner ring (GPU KV). Returns settled.
    function stepInnerEasing() {
      anim.cached += (target.cached - anim.cached) * 0.15;
      anim.delta += (target.delta - anim.delta) * 0.15;
      anim.gen += (target.gen - anim.gen) * 0.15;
      anim.total = target.total || 0;
      const settled = Math.abs(anim.cached - target.cached) < 0.5 &&
                      Math.abs(anim.delta - target.delta) < 0.5 &&
                      Math.abs(anim.gen - target.gen) < 0.5;
      if (settled) {
        anim.cached = target.cached;
        anim.delta = target.delta;
        anim.gen = target.gen;
      }
      return settled;
    }

    // Ring 2: checkpoints (the born/touch/death effects). Returns until.
    function paintRing2(cp, now) {
      let until = 0;
      const half = Math.max(0.006, 3 / R2); // N7: a label no thinner than 3px along the arc
      ctx.lineWidth = LW2;
      ctx.strokeStyle = '#151e2b';
      ctx.setLineDash(cp.state === 'na' ? [6, 6] : []);
      ctx.beginPath(); ctx.arc(CX, CY, R2, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      // The eviction wave: a 20 ms delay per slot by descending tokens
      const delayByTokens = {};
      if (fxEnabled) {
        cp.marks.filter(m => m.removedAt).sort((a, b) => b.tokens - a.tokens)
          .forEach((m, i) => { delayByTokens[m.tokens] = i * 20; });
      }
      for (const m of cp.marks) {
        if (m.frac == null) continue; // V1: the n_ctx axis is unconfirmed — the label is not placed
        const a = START + m.frac * Math.PI * 2;
        if (m.removedAt) {
          if (!fxEnabled) continue; // static mode: the dying ones are not shown
          const delay = delayByTokens[m.tokens] || 0;
          const death = clamp01((now - m.removedAt - delay) / FX.death); // 0→1
          const u = m.removedAt + FX.death + delay;
          if (u > until) until = u;
          if (death < 1) {
            ctx.globalAlpha = 1 - death;
            ctx.strokeStyle = '#64748b';
            ctx.lineWidth = LW2;
            // a fixed radius: the drift +6px at death gave the same
            // radial effect as the born pop; the fade — only via alpha
            ctx.beginPath(); ctx.arc(CX, CY, R2, a - half, a + half); ctx.stroke();
            ctx.globalAlpha = 1;
          }
          continue;
        }
        // a live label
        const f = fxEnabled ? fx(m, now) : { born: 0, touch: 0, until: 0 };
        if (f.until > until) until = f.until;
        const born = f.born, touch = f.touch;
        const r = R2; // a fixed radius: the +6px "pop" at born created
        // radial notches ("needles") on batch checkpoint creation;
        // the born flash stays (the white glow below), the radius — static
        ctx.globalAlpha = 1;
        ctx.strokeStyle = touch > 0 ? '#22c55e' : '#14b8a6';
        ctx.lineWidth = LW2;
        ctx.beginPath(); ctx.arc(CX, CY, r, a - half, a + half); ctx.stroke();
        if (born > 0) { // a white flash (two layers, no shadowBlur)
          ctx.globalAlpha = born;
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = LW2 + 4;
          ctx.beginPath(); ctx.arc(CX, CY, r, a - half, a + half); ctx.stroke();
        }
        if (touch > 0) { // a green restore glow (two layers)
          ctx.globalAlpha = touch * 0.55;
          ctx.strokeStyle = '#22c55e';
          ctx.lineWidth = LW2 + 6;
          ctx.beginPath(); ctx.arc(CX, CY, r, a - (half + 0.002), a + (half + 0.002)); ctx.stroke();
        }
        if (fxEnabled && m.touchedAt) { // a physical injector: a beam from Host RAM (R2) outward into VRAM (R)
          const beamAge = now - m.touchedAt;
          if (beamAge >= 0 && beamAge < FX.beam) {
            const ba = 1 - beamAge / FX.beam;
            // Vortex: the beam end in VRAM is shifted clockwise by BEAM_SWEEP —
            // the beam "unwinds" the outer orbit, it does not merely touch it.
            const aOut = a + BEAM_SWEEP;
            const rIn = R2 + LW2 / 2, rOut = R - LW / 2;
            // C1: a Host RAM (#14b8a6) → VRAM (#22c55e) gradient, fading in the stops
            const grad = ctx.createLinearGradient(
              CX + Math.cos(a) * rIn, CY + Math.sin(a) * rIn,
              CX + Math.cos(aOut) * rOut, CY + Math.sin(aOut) * rOut
            );
            grad.addColorStop(0, 'rgba(20, 184, 166, ' + (ba * 0.9) + ')');
            grad.addColorStop(1, 'rgba(34, 197, 94, 0)');
            ctx.strokeStyle = grad;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(CX + Math.cos(a) * rIn, CY + Math.sin(a) * rIn);
            ctx.lineTo(CX + Math.cos(aOut) * rOut, CY + Math.sin(aOut) * rOut);
            ctx.stroke();
            ctx.globalAlpha = 1;
          }
        }
        ctx.globalAlpha = 1;
      }
      if (cp.state === 'na') {
        ctx.fillStyle = '#64748b';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('N/A', CX, CY - R2);
      }
      return until;
    }

    // Ring 3: the prompt cache (the born-grow/touch/death effects). Returns until.
    function paintRing3(ram, now) {
      let until = 0;
      ctx.lineWidth = LW3;
      ctx.strokeStyle = '#151e2b';
      ctx.setLineDash(ram.state === 'na' ? [6, 6] : []);
      ctx.beginPath(); ctx.arc(CX, CY, R3, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      // The layout (B2): the dying arc shrinks in place, the live ones slide
      // into the freed spot — no overlaps. ring3Layout — for hover.
      const layout = computeRing3Layout(ram, now, fxEnabled);
      ring3Layout = layout.arcs;
      if (layout.until > until) until = layout.until;
      for (const seg of layout.arcs) {
        if (seg.len * R3 * Math.PI * 2 < MIN_ARC_PX) continue; // the same invariant as the inner ring
        const a0 = START + seg.start * Math.PI * 2;
        const a1 = a0 + seg.len * Math.PI * 2;
        if (seg.removedAt) {
          // dying: a gray-red 2 s fade
          ctx.globalAlpha = 1 - seg.death;
          ctx.strokeStyle = '#f43f5e';
          ctx.lineWidth = LW3;
          ctx.beginPath(); ctx.arc(CX, CY, R3, a0, a1); ctx.stroke();
          ctx.globalAlpha = 1;
          continue;
        }
        const f = fxEnabled ? fx(seg, now) : { touch: 0, until: 0 };
        // a cache record born: the arc grows 0→len over FX.grow, then the FX.born flash
        const bornFlash = fxEnabled ? decay((now - (seg.createdAt + FX.grow)), FX.born) : 0;
        const touch = f.touch;
        ctx.globalAlpha = 1;
        // touch = "updated" — a white-blue flash (green — only checkpoint restores)
        ctx.strokeStyle = touch > 0 ? '#7dd3fc' : '#38bdf8';
        ctx.lineWidth = LW3;
        ctx.beginPath(); ctx.arc(CX, CY, R3, a0, a1); ctx.stroke();
        if (bornFlash > 0) {
          ctx.globalAlpha = bornFlash;
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = LW3 + 4;
          ctx.beginPath(); ctx.arc(CX, CY, R3, a0, a1); ctx.stroke();
        }
        if (touch > 0) {
          ctx.globalAlpha = touch * 0.55;
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = LW3 + 6;
          ctx.beginPath(); ctx.arc(CX, CY, R3, a0, a1); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      if (ram.state === 'na') {
        ctx.fillStyle = '#64748b';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('N/A', CX, CY - R3);
      }
      return until;
    }

    // The painter: DOM-independent, receives now. Returns until (the end of the effects).
    function paintAll(model, now) {
      let until = 0;
      ctx.clearRect(0, 0, 480, 480);
      // 1. The inner cache rings (the R3 prompt cache, the R2 checkpoints with beams)
      until = Math.max(until, paintRing3(model.ram, now));
      until = Math.max(until, paintRing2(model.cp, now));
      // 2. The outer ring (the GPU VRAM KV slot) — on top, the basic dark backing
      ctx.lineWidth = LW;
      ctx.strokeStyle = '#151e2b';
      ctx.beginPath(); ctx.arc(CX, CY, R, 0, Math.PI * 2); ctx.stroke();
      const tot = anim.total > 0 ? anim.total : 1; // V1: total=0 (unconfirmed) — zero arcs
      const cachedAngle = (anim.cached / tot) * (Math.PI * 2);
      const deltaAngle = (anim.delta / tot) * (Math.PI * 2);
      const genAngle = (anim.gen / tot) * (Math.PI * 2);
      // MIN_ARC_PX — the VRAM ring invariant: a subpixel arc
      // at LW=26 turns into a radial "needle" via the stroke
      if (anim.cached > 2 && cachedAngle * R >= MIN_ARC_PX) { ctx.strokeStyle = '#22c55e'; ctx.beginPath(); ctx.arc(CX, CY, R, START, START + cachedAngle); ctx.stroke(); }
      if (anim.delta > 2 && deltaAngle * R >= MIN_ARC_PX) {
        const sA = START + cachedAngle, eA = sA + deltaAngle;
        ctx.strokeStyle = '#eab308';
        ctx.beginPath(); ctx.arc(CX, CY, R, sA, eA); ctx.stroke();
      }
      // Decode — a clean arc without dots/balls: a subpixel arc at LW=26
      // is drawn as a radial "needle", so shorter than MIN_ARC_PX we do not pass to Canvas
      if (anim.gen > 2 && genAngle * R >= MIN_ARC_PX) {
        const sA = START + cachedAngle + deltaAngle, eA = sA + genAngle;
        ctx.strokeStyle = '#a855f7';
        ctx.beginPath(); ctx.arc(CX, CY, R, sA, eA); ctx.stroke();
      }
      return until;
    }

    // The controller: spin while !settled || now < until. In idle — stop.
    function frame() {
      const now = Date.now() + clockOffset;
      const settled = stepInnerEasing();
      const until = paintAll(ringModel, now);
      if (settled && now >= until) { animRunning = false; return; }
      requestAnimationFrame(frame);
    }
    kick();

    // R5: hover hints on the canvas: radius -> ring, angle -> entity.
    const tooltip = document.getElementById('ring-tooltip');
    function angleDiff(a, b) {
      let d = Math.abs(a - b) % (Math.PI * 2);
      return d > Math.PI ? Math.PI * 2 - d : d;
    }
    function fmtTime(ts) { return ts ? new Date(ts).toLocaleTimeString() : '—'; }
    cvs.addEventListener('pointermove', (ev) => {
      const rect = cvs.getBoundingClientRect();
      const x = (ev.clientX - rect.left) * (cvs.width / rect.width);
      const y = (ev.clientY - rect.top) * (cvs.height / rect.height);
      const dist = Math.hypot(x - CX, y - CY);
      const ang = Math.atan2(y - CY, x - CX);
      let html = null;
      if (dist >= R2 - LW2 / 2 - 3 && dist <= R2 + LW2 / 2 + 3) {
        // Ring 2: the nearest live label within 0.03 rad (N7: the threshold under the half width)
        const now = Date.now() + clockOffset;
        let best = null, bestD = 0.03;
        for (const m of ringModel.cp.marks) {
          if (m.removedAt && now - m.removedAt > FX.death) continue; // dead labels are not shown
          const d = angleDiff(ang, START + m.frac * Math.PI * 2);
          if (d <= bestD) { bestD = d; best = m; }
        }
        if (best) {
          html = '<b>' + best.tokens.toLocaleString() + ' ' + t('tok') + '</b> · ' +
            Number(best.sizeMiB).toFixed(1) + ' MiB · ' + t('tt_cp_created') + ' ' + fmtTime(best.createdAt) +
            (best.restoreCount ? ' · ' + t('tt_cp_restored') + best.restoreCount : '') +
            (best.removedAt ? ' · ' + t('tt_cp_evicted') + ' ' + fmtTime(best.removedAt) : '') +
            '<div class="tt-dim">' + t('tt_cp_partial') + '</div>';
        }
      } else if (dist >= R3 - LW3 / 2 - 3 && dist <= R3 + LW3 / 2 + 3) {
        // Ring 3: a segment from the last layout (the live arcs), the nearest within 0.05 rad
        let best = null, bestD = 0.05;
        for (const seg of ring3Layout) {
          // the same invariant as in paintRing3: the tooltip is not shown
          // for segments that are not drawn (sub-pixel)
          if (seg.removedAt || seg.len * R3 * Math.PI * 2 < MIN_ARC_PX) continue;
          const a0 = START + seg.start * Math.PI * 2;
          const arcLen = seg.len * Math.PI * 2;
          let rel = (ang - a0) % (Math.PI * 2);
          if (rel < 0) rel += Math.PI * 2;
          const d = rel <= arcLen ? 0 : Math.min(rel - arcLen, Math.PI * 2 - rel);
          if (d <= bestD) { bestD = d; best = seg; }
        }
        if (best) {
          html = '<b>' + best.tokens.toLocaleString() + ' ' + t('tok') + '</b> · ' + t('tt_cp_count') + ' ' +
            best.checkpoints + ' · ' + Number(best.sizeMiB).toFixed(1) + ' MiB';
        }
      }
      if (html) {
        tooltip.innerHTML = html;
        tooltip.style.display = 'block';
        let left = ev.clientX - rect.left + 18;
        if (left + tooltip.offsetWidth > rect.width) left = ev.clientX - rect.left - tooltip.offsetWidth - 18;
        let top = ev.clientY - rect.top - 10;
        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
      } else {
        tooltip.style.display = 'none';
      }
    });
    cvs.addEventListener('pointerleave', () => { tooltip.style.display = 'none'; });

    function esc(v) {
      return String(v ?? '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[c]));
    }
    function num(v, def = 0) { const n = Number(v); return Number.isFinite(n) ? n : def; }

    let lastEventsHash = '';
    let lastHistHash = '';
    let lastLiveHash = '';
    // C1: the client cache of events/history — the server sends the arrays only on
    // a version change (eventsRev/historyRev), otherwise [] arrives
    let clientEvents = [], clientHistory = [], clientEventsRev = -1, clientHistoryRev = -1;

    function focusTask(taskId) {
      if (!taskId) return;
      const card = document.getElementById('task-card-' + taskId);
      const row = document.getElementById('hist-row-' + taskId);
      
      document.querySelectorAll('.hist-table tbody tr').forEach(r => r.classList.remove('hist-row-highlight'));
      if (row) row.classList.add('hist-row-highlight');

      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        card.classList.remove('highlight-pulse');
        void card.offsetWidth;
        card.classList.add('highlight-pulse');
        card.addEventListener('animationend', () => {
          card.classList.remove('highlight-pulse');
        }, { once: true });
      }
    }

    document.getElementById('log-feed').addEventListener('click', (e) => {
      const header = e.target.closest('.task-card-header');
      if (header && header.dataset.taskId) {
        focusTask(header.dataset.taskId);
      }
    });

    document.getElementById('hist-rows').addEventListener('click', (e) => {
      const row = e.target.closest('tr[data-task-id]');
      if (row && row.dataset.taskId) {
        focusTask(row.dataset.taskId);
      }
    });
    document.getElementById('hist-live-row').addEventListener('click', (e) => {
      const row = e.target.closest('tr[data-task-id]');
      if (row && row.dataset.taskId) {
        focusTask(row.dataset.taskId);
      }
    });

    // EVENT LOG: collapsing the right column into a narrow vertical tab
    document.getElementById('btn-toggle-log').addEventListener('click', () => {
      document.querySelector('.main-layout').classList.toggle('log-collapsed');
    });
    document.getElementById('log-collapsed-bar').addEventListener('click', () => {
      document.querySelector('.main-layout').classList.remove('log-collapsed');
    });

    // The filter popover control
    const btnFilter = document.getElementById('btn-log-filter');
    const popover = document.getElementById('log-filter-popover');
    const feed = document.getElementById('log-feed');
    const chkTime = document.getElementById('filter-show-time');

    btnFilter.addEventListener('click', (e) => {
      e.stopPropagation();
      popover.style.display = popover.style.display === 'none' ? 'block' : 'none';
    });
    document.addEventListener('click', (e) => {
      if (!popover.contains(e.target) && e.target !== btnFilter) {
        popover.style.display = 'none';
      }
    });

    // The time toggle
    chkTime.addEventListener('change', () => {
      feed.classList.toggle('show-time', chkTime.checked);
      localStorage.setItem('monitor_show_time', chkTime.checked ? '1' : '0');
    });
    if (localStorage.getItem('monitor_show_time') === '1') {
      chkTime.checked = true;
      feed.classList.add('show-time');
    }

    // The category checkboxes
    document.querySelectorAll('.tag-filter').forEach(chk => {
      const target = chk.dataset.target;
      const storageKey = 'monitor_filter_' + target;

      chk.addEventListener('change', () => {
        feed.classList.toggle('hide-' + target, !chk.checked);
        localStorage.setItem(storageKey, chk.checked ? '1' : '0');
      });

      // Restoring from localStorage
      if (localStorage.getItem(storageKey) === '0') {
        chk.checked = false;
        feed.classList.add('hide-' + target);
      }
    });

    // The "Reset / Select all" button
    document.getElementById('btn-filter-reset').addEventListener('click', () => {
      document.querySelectorAll('.tag-filter').forEach(chk => {
        chk.checked = true;
        feed.classList.remove('hide-' + chk.dataset.target);
        localStorage.removeItem('monitor_filter_' + chk.dataset.target);
      });
    });

    function renderState(s) {
      const t = s.task || {};
      const ram = s.ramCache || {};
      const totalCtx = num(s.n_ctx, 0); // V1: 0 = unconfirmed, we show "—"
      const stage = s.currentStage || 'IDLE';
      // Provenance labels: the source of the cached prefix (NOT derived from cached n_tokens)
      const SRC_LABEL = { 'hot-vram': 'HOT VRAM', 'prompt-cache-ram': 'RAM CACHE', 'empty': 'COLD' };

      // Two-level status: level 1 — the connection to the server, level 2 — the task phase
      // (the elements may be absent in the header — safe checks)
      const sDot = document.getElementById('tier-dot');
      const sServerText = document.getElementById('tier-server-status');
      const b = document.getElementById('status-badge');
      if (sDot && sServerText) {
        if (s.mode === 'PLAYBACK') {
          // PLAYBACK: the server status is not shown as LIVE — a yellow dot + a badge
          sDot.className = 'status-dot dot-playback';
          sServerText.innerText = 'PLAYBACK';
        } else if (s.serverOnline) {
          sDot.className = 'status-dot dot-online';
          sServerText.innerText = 'ONLINE';
        } else {
          sDot.className = 'status-dot dot-offline';
          sServerText.innerText = 'OFFLINE';
        }
      }
      // The compact badge always holds the current stage (IDLE, PREFILL, DECODE, DONE):
      // PLAYBACK does not depend on the llama-server availability — OFFLINE is not shown
      if (b) {
        const online = s.serverOnline || s.mode === 'PLAYBACK';
        b.className = 'badge-mini-status ' + (online ? 'badge-on phase-' + stage : 'badge-off');
        b.innerText = online ? (s.is_processing ? stage : 'IDLE') : 'OFFLINE';
      }
      // EVENT LOG: the activity badge on the collapsed vertical tab
      const logBadge = document.getElementById('log-bar-badge');
      if (logBadge) logBadge.innerText = String(clientEvents.length);
      // /metrics: the harness queue (deferred tasks) — the badge only when > 0
      // #18: and only if /metrics REALLY answered (metricsSeen) — otherwise
      // the factory zeros would look like a "queue of 0" with an unavailable source
      const sm = s.serverMetrics || {};
      const qd = sm.metricsSeen ? num(sm.queueDeferred, 0) : 0;
      const qb = document.getElementById('queue-badge');
      if (qb) {
        qb.style.display = qd > 0 ? 'block' : 'none';
        if (qd > 0) document.getElementById('queue-val').innerText = qd;
      }
      // A1: the player — in LIVE the speed/"again" are inactive (the "Log" button is always active);
      // a pause in non-LIVE is synchronized in onSnapshot (from s.player.paused),
      // the render does not block. "⏭ TO LIVE" is visible only in PLAYBACK
      if (ctlSpeed) ctlSpeed.disabled = (s.mode === 'LIVE');
      if (ctlRestart) ctlRestart.disabled = (s.mode === 'LIVE');
      if (ctlLive) ctlLive.style.display = (s.mode === 'PLAYBACK' && s.liveCapable) ? 'inline-flex' : 'none';
      document.getElementById('hdr-ctx').innerText = totalCtx > 0 ? totalCtx.toLocaleString() : '—';

      target.total = totalCtx;
      const inputVal = num(t.totalPrompt, 0);
      // DONE: the donut and the legend keep the detailed anatomy of the last request
      // (reuse/prefill/decode separately). IDLE: the whole finished request physically
      // settled into VRAM as a cache — the ring and the legend merge into a single green KV sector
      // (the anatomy is not lost: it is fixed in the REQUEST HISTORY line).
      const cachedVal = num(t.kvRestoredTokens, 0);
      const deltaVal = num(t.deltaTokens, 0);
      const genVal = num(t.generatedTokens, 0);
      const idleMerge = stage === 'IDLE' && (cachedVal + deltaVal + genVal) > 0;
      // Fork A: on a context shift (truncated=1) the physical remainder in the slot
      // = finalSlotTokens (n_tokens from stop processing). The donut = a mirror of the VRAM buffer:
      // the base strictly switches to finalSlotTokens, the pre-shift anatomy no longer
      // reflects the real KV content → a single green sector (like the IDLE merge).
      const finalVal = num(t.finalSlotTokens, 0);
      const isTrunc = !!t.isTruncated && finalVal > 0;
      // LIVE: /slots.n_prompt_tokens = prompt.tokens.size() — the authoritative
      // physical slot size. Before a context shift it is ≈ cached+delta+gen; after
      // the shift (seq_rm) the anatomy no longer reflects the KV → base = n_prompt_tokens,
      // a merge into a single green (like TRUNC). A 2%/64 tok tolerance — protection
      // against polling chatter (n_prompt_tokens vs n_decoded are updated separately).
      const liveNpt = (s.mode === 'LIVE' && s.is_processing) ? num(s.n_prompt_tokens, 0) : 0;
      const anatomySum = cachedVal + deltaVal + genVal;
      const liveShift = liveNpt > 0 && liveNpt < anatomySum - Math.max(64, anatomySum * 0.02);
      const merge = idleMerge || isTrunc || liveShift;

      // M3: SAVED TIME — only with a MEASURED prefill speed (no magic
      // constants). The estimate: the restored tokens did not pass the prefill, the time
      // is counted by the last measured t/s (the current task or past ones).
      const roiTps = num(t.prefillTps, 0) > 0 ? num(t.prefillTps, 0) : num(s.lastPrefillTps, 0);
      const roiRow = document.getElementById('row-kv-roi');
      const lblRoi = document.getElementById('lbl-kv-roi');
      if (roiRow && lblRoi) {
        if (cachedVal > 0 && roiTps > 0) {
          lblRoi.innerText = '~' + (cachedVal / roiTps).toFixed(1) + 's @ ' + roiTps.toFixed(0) + ' t/s';
        } else if (cachedVal > 0) {
          lblRoi.innerText = cachedVal.toLocaleString() + ' ' + window.__wwwT('tok_wait');
        } else {
          lblRoi.innerText = '—';
        }
      }

      // LIVE: base = /slots.n_prompt_tokens (the authoritative slot size);
      // TRUNC: base = finalSlotTokens (the real remainder after the shift); otherwise the anatomy sum
      const rawUsed = liveNpt > 0 ? liveNpt : (isTrunc ? finalVal : anatomySum);
      // IDLE/TRUNC merge: yellow (prefill) and violet (decode) flow into the green KV
      const mCached = merge ? rawUsed : cachedVal;
      const mDelta = merge ? 0 : deltaVal;
      const mGen = merge ? 0 : genVal;

      if (totalCtx > 0 && rawUsed > totalCtx && rawUsed > 0) {
        const scale = totalCtx / rawUsed;
        target.cached = mCached * scale;
        target.delta = mDelta * scale;
        target.gen = mGen * scale;
      } else {
        target.cached = mCached;
        target.delta = mDelta;
        target.gen = mGen;
      }
      // kick only on a real divergence: the P1 tick calls renderState once per
      // second, and an unconditional kick would spin drawReactor in idle (the P5 gate)
      if (Math.abs(anim.cached - target.cached) >= 0.5 ||
          Math.abs(anim.delta  - target.delta)  >= 0.5 ||
          Math.abs(anim.gen    - target.gen)    >= 0.5 ||
          anim.total !== target.total) kick();

      // The outer rings (R3/R4): the model is rebuilt every tick, kick on
      // a change (hash) or while the effects are active (until > now). In idle
      // (no changes, until < now) the frame does not spin (the P5 gate).
      const model = buildRingModel(s);
      const modelHash = JSON.stringify(model);
      const modelChanged = modelHash !== lastRingModelHash;
      if (modelChanged) {
        lastRingModelHash = modelHash;
        ringModel = model;
      }
      if (modelChanged || (fxEnabled && model.until > Date.now() + clockOffset)) kick();

      const totalUsed = rawUsed; // LIVE: n_prompt_tokens; TRUNC: finalSlotTokens; otherwise the anatomy
      const usedPct = totalCtx > 0 ? ((totalUsed / totalCtx) * 100).toFixed(1) : '0.0';

      const coreBg = document.getElementById('donut-core-bg');
      const corePhase = document.getElementById('core-phase');
      const coreSpeed = document.getElementById('core-speed');
      
      corePhase.className = 'phase-badge phase-' + stage;

      // M4: the hardware bottleneck of the phase (pure UI, no backend)
      const bn = document.getElementById('core-bottleneck');
      if (stage === 'PREFILL') bn.innerText = 'PREFILL / BATCH COMPUTE';
      else if (stage === 'DECODE') bn.innerText = 'DECODE / TOKEN STEP';
      else bn.innerText = 'IDLE';

      document.getElementById('row-reuse').classList.toggle('active-row', stage === 'PREFIX' || stage === 'KV_RESTORE');
      document.getElementById('row-prefill').classList.toggle('active-row', stage === 'PREFILL');
      document.getElementById('row-decode').classList.toggle('active-row', stage === 'DECODE');

      if (stage === 'DECODE') {
        coreBg.classList.remove('breathe-idle');
        corePhase.innerText = 'DECODE';
        coreSpeed.innerText = t.decodeTps ? num(t.decodeTps).toFixed(1) + ' t/s' : '...';
        coreSpeed.style.color = 'var(--decode)';
      } else if (stage === 'PREFILL') {
        coreBg.classList.remove('breathe-idle');
        corePhase.innerText = 'PREFILL';
        const prg = t.prefillProgress ? t.prefillProgress + '% ' : '';
        coreSpeed.innerText = prg + (t.prefillTps ? num(t.prefillTps).toFixed(0) + ' t/s' : '...');
        coreSpeed.style.color = 'var(--prefill)';
      } else if (stage === 'PREFIX' || stage === 'KV_RESTORE') {
        coreBg.classList.remove('breathe-idle');
        corePhase.innerText = 'KV REUSE';
        coreSpeed.innerText = cachedVal.toLocaleString() + ' ' + window.__wwwT('tok');
        coreSpeed.style.color = 'var(--reuse)';
      } else if (stage === 'INPUT') {
        coreBg.classList.remove('breathe-idle');
        corePhase.innerText = 'INPUT';
        coreSpeed.innerText = inputVal.toLocaleString() + ' ' + window.__wwwT('tok');
        coreSpeed.style.color = 'var(--input)';
      } else if (stage === 'DONE') {
        coreBg.classList.remove('breathe-idle');
        corePhase.innerText = 'DONE ✓';
        coreSpeed.innerText = t.decodeTps ? num(t.decodeTps).toFixed(1) + ' t/s' : '—';
        coreSpeed.style.color = '#94a3b8';
      } else {
        coreBg.classList.add('breathe-idle');
        corePhase.innerText = 'IDLE';
        coreSpeed.innerText = '—';
        coreSpeed.style.color = '#fff';
      }

      document.getElementById('core-pct').innerText = 'CONTEXT: ' + totalUsed.toLocaleString() + ' (' + usedPct + '%)';
      document.getElementById('val-input').innerText = inputVal.toLocaleString();
      // IDLE/TRUNC legend merge: in sync with the ring — the whole volume in KV REUSE
      document.getElementById('val-cached').innerHTML = merge
        ? totalUsed.toLocaleString()
        : cachedVal.toLocaleString() +
          (t.slotSource ? ' <span style="color:var(--dim);font-size:10px">' + (SRC_LABEL[t.slotSource] || t.slotSource) + '</span>' : '') +
          (t.checkpointRestore ? ' <span style="color:var(--kv-save);font-size:10px" title="' + window.__wwwT('tt_cp_restored_vram') + '">✚CP</span>' : '');
      document.getElementById('val-delta').innerText = (merge ? 0 : deltaVal).toLocaleString();
      document.getElementById('val-gen').innerText = (merge ? 0 : genVal).toLocaleString();
      document.getElementById('val-free').innerText = totalCtx > 0 ? Math.max(0, totalCtx - totalUsed).toLocaleString() + ' ' + window.__wwwT('tok_unit') : '—';

      // R6: legend of the outer rings (ring 2 — checkpoints, ring 3 — prompt cache)
      document.getElementById('legend-cp').innerText = num(ram.cpCurrent, 0) + '/' + (ram.cpMax == null ? '—' : ram.cpMax) +
        ' (+' + num(ram.cpCreated, 0) + ' / −' + num(ram.cpErased, 0) + ')';
      const legendRam = (ram.limitMiB != null)
        ? num(ram.promptCacheMiB, 0).toLocaleString() + ' / ' + (ram.limitMiB > 0 ? num(ram.limitMiB).toLocaleString() + ' MiB' : '∞') + ' · ' +
          num(ram.promptsCount, 0).toLocaleString() + ' prompts · ' +
          (ram.entries || []).filter(e => !e.removedAt).reduce((a, e) => a + num(e.tokens, 0), 0).toLocaleString() + ' / ' +
          // the EFFECTIVE token limit (upstream est = max(configured, limit_size/size_per_token));
          // without est (the old format) — the configured one
          (ram.limitTokensEff != null ? num(ram.limitTokensEff).toLocaleString()
            : (ram.limitTokens != null ? num(ram.limitTokens).toLocaleString() : '—')) + ' ' + window.__wwwT('tok')
        : 'N/A';
      document.getElementById('legend-ram').innerText = legendRam +
        ' · [' + (ram.stateAt ? new Date(ram.stateAt).toLocaleTimeString() : '—') + ']';

      // DRAFT-MTP ACC PER POS: the bars are built ONLY from state.specMtp.posAcc
      // (session cumulative, "statistics draft-mtp: #acc rate/pos").
      // The expected number of positions — from the real launch configuration
      // (specConfig.draftNMax, the startup log "- n_max=N"); no caps and no fixed
      // P1–P5/P1–P12. state.task.specPosAcc (SPEC TOTAL) is NOT used
      // for the MTP bars.
      document.getElementById('lbl-mtp-step').innerText = t.specMeanLen ? num(t.specMeanLen).toFixed(2) + ' ' + window.__wwwT('tok_step') : window.__wwwT('tok_per_step');
      const mtp = Array.isArray(s.specMtp.posAcc) ? s.specMtp.posAcc : [];
      const configuredN = Number.isInteger(s.specConfig.draftNMax) ? s.specConfig.draftNMax : null;
      const observedN = mtp.length;
      const mtpN = configuredN != null ? Math.max(configuredN, observedN) : observedN;
      if (mtpN !== renderedMtpN) buildMtpBars(mtpN);
      const lblMtpPos = document.getElementById('lbl-mtp-pos');
      if (lblMtpPos) lblMtpPos.innerText = mtpN > 0 ? 'DRAFT-MTP ACC PER POS (1–' + mtpN + '):' : 'DRAFT-MTP ACC PER POS:';
      for (let i = 0; i < renderedMtpN; i++) {
        const f = document.getElementById('mtp-f' + i);
        const v = document.getElementById('mtp-v' + i);
        if (f && v) {
          const val = mtp[i];
          if (val == null) {
            f.style.height = '0%';
            v.innerText = '—';
          } else {
            const n = Math.max(0, Math.min(100, num(val, 0)));
            f.style.height = n + '%';
            v.innerText = n + '%';
          }
        }
      }

      // SPECULATIVE — SESSION CUMULATIVE (⚠ = the mean = 1 + Σ rates invariant is violated)
      const specCumText = (o) => (o.accTokens > 0 || o.genTokens > 0)
        ? num(o.accTokens).toLocaleString() + '/' + num(o.genTokens).toLocaleString() + ' ' + window.__wwwT('tok_unit') + ' (' + (o.genTokens > 0 ? (num(o.accTokens) / num(o.genTokens) * 100).toFixed(1) : '0.0') + '%) | mean ' + num(o.meanLen).toFixed(2) + (o.invariantOk === false ? ' ⚠' : '')
        : '—';
      const lblSpecTotal = document.getElementById('lbl-spec-total');
      if (lblSpecTotal) {
        lblSpecTotal.innerText = (t.specAccepted != null)
          ? num(t.specAccepted).toLocaleString() + '/' + num(t.specGenerated).toLocaleString() + ' ' + window.__wwwT('tok_unit') + ' (' + t.specAcceptance + '%) | mean ' + num(t.specMeanLen).toFixed(2) + (t.specInvariantOk === false ? ' ⚠' : '')
          : '—';
      }
      const lblSpecMtp = document.getElementById('lbl-spec-mtp');
      if (lblSpecMtp) lblSpecMtp.innerText = specCumText(s.specMtp || {});
      const lblNg = document.getElementById('lbl-ngram');
      if (lblNg) lblNg.innerText = specCumText(s.ngram || {});

      const events = clientEvents; // C1: the cache (the server sends the array only on an eventsRev change)
      const currentEventsHash = (s.eventsRev || 0) + '|' + events.map(e => (e.id || '') + (e.stage || '')).join('|') + stage; // B2
      if (currentEventsHash !== lastEventsHash) {
        lastEventsHash = currentEventsHash;
        const feed = document.getElementById('log-feed');
        const isPinnedToTop = feed.scrollTop <= 30;
        const oldScrollHeight = feed.scrollHeight;
        const oldScrollTop = feed.scrollTop;
        const groups = {};
        const groupOrder = [];

        for (const ev of events) {
          const tid = ev.taskId || 'SYS';
          if (!groups[tid]) {
            groups[tid] = [];
            groupOrder.push(tid);
          }
          groups[tid].push(ev);
        }

        let html = '';
        for (const tid of groupOrder) {
          const evList = groups[tid];
          const isCurrent = (t.id != null && tid === String(t.id));
          const histItem = clientHistory.find(h => String(h.id) === String(tid));
          let statusText = 'DONE ✓';
          let statusColor = '#94a3b8';
          let summary = '';

          if (isCurrent && s.is_processing) {
            statusText = stage;
            statusColor = (stage === 'DECODE' ? 'var(--decode)' : stage === 'PREFILL' ? 'var(--prefill)' : 'var(--input)');
            summary = 'INPUT: ' + (t.totalPrompt||0).toLocaleString() + ' | GEN: ' + (t.generatedTokens||0).toLocaleString() + ' (' + (t.decodeTps||0) + ' t/s)';
          } else if (histItem) {
            if (histItem.cancelled) {
              statusText = 'CANCELLED ✗';
              statusColor = 'var(--red)';
            } else if (histItem.isTruncated) {
              if (histItem.ctxShifted) {
                statusText = 'SHIFTED ↺';
                statusColor = 'var(--prefill)';
              } else {
                statusText = 'OVERFLOW ✗';
                statusColor = 'var(--red)';
              }
            } else {
              statusText = 'DONE ✓';
              statusColor = '#94a3b8';
            }
            summary = 'INPUT: ' + num(histItem.input).toLocaleString() + ' | REUSE: ' + num(histItem.kvReuse).toLocaleString() + ' | ' + num(histItem.speed).toFixed(1) + ' t/s';
          } else if (tid === 'SYS') {
            statusText = 'SYSTEM';
            statusColor = 'var(--dim)';
            summary = window.__wwwT('sys_events');
          }

          html += '<div class="task-card ' + (isCurrent && s.is_processing ? 'active-task-card' : '') + '" id="task-card-' + esc(tid) + '" data-task-id="' + esc(tid) + '">';
          html += '  <div class="task-card-header" data-task-id="' + esc(tid) + '">';
          html += '    <div class="task-card-title">';
          html += '      <span class="task-badge-id">' + (tid === 'SYS' ? 'SYSTEM' : '#' + esc(tid)) + '</span>';
          html += '      <span class="task-card-summary">' + esc(summary) + '</span>';
          html += '    </div>';
          html += '    <div style="font-weight:bold; color:' + statusColor + ';">' + esc(statusText) + '</div>';
          html += '  </div>';
          html += '  <div class="task-card-body">';

          for (const e of evList) {
            // i18n: the server sends the dictionary keys + params; sourceKey/reasonKey
            // are resolved into the current language BEFORE substitution into the detail template
            // window.__wwwT: in renderState t is the task object (a shadow of the i18n function)
            const ep = e.params || {};
            const resolved = Object.assign({}, ep);
            if (ep.sourceKey) resolved.source = window.__wwwT(ep.sourceKey);
            if (ep.reasonKey) resolved.reason = window.__wwwT(ep.reasonKey);
            html += '    <div class="log-row ' + esc(e.stage) + '" data-tag="' + esc(e.tag) + '">';
            html += '      <span class="log-time">[' + esc(e.ts) + ']</span>';
            html += '      <span class="tag-pill log-tag">' + esc(e.tag) + '</span>';
            html += '      <span class="log-msg"><b>' + esc(window.__wwwT(e.titleKey, ep)) + ':</b> ' + window.__wwwT(e.detailKey, resolved) + '</span>';
            html += '    </div>';
          }

          html += '  </div>';
          html += '</div>';
        }

        feed.innerHTML = html;
        if (isPinnedToTop) {
          feed.scrollTop = 0;
        } else {
          feed.scrollTop = oldScrollTop + (feed.scrollHeight - oldScrollHeight);
        }
      }

      const history = clientHistory; // C1: the cache (the server sends the array only on a historyRev change)
      // C2: the live line — a separate tbody with its own gate (activeTaskHash changes
      // every decode tick and used to force a rebuild of the whole table)
      const alreadyInHistory = history.length > 0 && String(history[0].id) === String(t.id);
      const liveActive = s.is_processing && t.id && !alreadyInHistory && stage !== 'DONE';
      const liveKey = liveActive
        ? JSON.stringify([t.id, t.totalPrompt, t.kvRestoredTokens, t.lcpTokens, t.slotSource, t.checkpointRestore, t.reprocessReason, t.prefillTokens, t.deltaTokens, t.generatedTokens, t.decodeTps, t.prefillTps, t.specAccepted, t.specGenerated, t.specAcceptance, s._processingStart, stage])
        : '';
      if (liveKey !== lastLiveHash) {
        lastLiveHash = liveKey;
        let liveRowHtml = '';
        if (liveActive) {
          const liveDecodeSpeed = (stage === 'DECODE' && num(t.decodeTps) > 0)
            ? ' <span style="font-size:9.5px;color:var(--dim)">(' + num(t.decodeTps).toFixed(1) + ' t/s)</span>'
            : '';
          // The live SKEW is meaningful only after the KV line (cached/restored)
          // or the full re-processing verdict (the KV decision is already known: 0)
          const kvKnown = Boolean(t.cachedFirstSeen || t.checkpointRestore || t.reprocessReason);
          // FULL_REPROCESS: the whole prompt was recomputed — SKEW does not exist (like a cold start)
          const cpSkewLive = (kvKnown && !t.reprocessReason) ? Math.max(0, num(t.lcpTokens) - num(t.kvRestoredTokens)) : 0;
          const r = histCells({
            input: t.totalPrompt, kvReuse: t.kvRestoredTokens, lcp: t.lcpTokens,
            slotSource: t.slotSource, checkpointRestore: t.checkpointRestore,
            reprocessReason: t.reprocessReason,
            cpSkew: cpSkewLive,
            divergeInfo: t.divergeInfo, kvPending: !kvKnown,
            prefill: t.prefillTokens || t.deltaTokens,
            prefillSpeed: t.prefillTps || s.lastPrefillTps || null,
            prefillSource: t.prefillTokens ? 'measured' : 'estimated'
          }, history[0]);
          liveRowHtml =
            '<tr class="hist-row-live" id="hist-row-' + esc(t.id) + '" data-task-id="' + esc(t.id) + '"' + r.titleAttr + '>' +
            '  <td style="color:var(--dim);font-size:10px;font-variant-numeric:tabular-nums">' + fmtTime(s._processingStart) + '</td>' +
            '  <td><b style="color:var(--input)">#' + esc(t.id) + '</b></td>' +
            '  <td style="color:var(--input)">' + num(t.totalPrompt).toLocaleString() + '</td>' +
            '  <td style="color:var(--reuse)">' + num(t.kvRestoredTokens).toLocaleString() + '</td>' +
            '  <td>' + r.effHtml + '</td>' +
            '  <td>' + r.lostHtml + '</td>' +
            '  <td>' + r.statusHtml + '</td>' +
            '  <td>' + r.skewHtml + '</td>' +
            '  <td style="color:var(--prefill)">' + r.prefillHtml + '</td>' +
            '  <td style="color:var(--decode)">' + num(t.generatedTokens).toLocaleString() + liveDecodeSpeed + '</td>' +
            '  <td style="color:var(--mtp);font-size:10px">' + (t.specAccepted != null ? num(t.specAccepted).toLocaleString() + '/' + num(t.specGenerated).toLocaleString() + ' (' + t.specAcceptance + '%)' : '...') + '</td>' +
            '</tr>';
        }
        document.getElementById('hist-live-row').innerHTML = liveRowHtml;
      }
      // C2: the history table — a rebuild only on a change of the history itself
      const histDataHash = history.map(h => (h.id + h.time + (h.specAccepted || '') + (h.specMeanLen || '') + (h.cancelled ? 'c' : ''))).join('|');
      if (histDataHash !== lastHistHash) {
        lastHistHash = histDataHash;
        document.getElementById('hist-rows').innerHTML = history.map((h, idx) => {
          const r = histCells(h, history[idx + 1]);
          const decodeSpeedStr = h.cancelled
            ? ' <span style="color:var(--red);font-size:9.5px;">(✗)</span>'
            : (num(h.speed) > 0 ? ' <span style="font-size:9.5px;color:var(--dim)">(' + num(h.speed).toFixed(1) + ' t/s)</span>' : '');
          return '<tr id="hist-row-' + esc(h.id) + '" data-task-id="' + esc(h.id) + '"' + r.titleAttr + '>' +
            '  <td style="color:var(--dim);font-size:10px;font-variant-numeric:tabular-nums">' + esc(h.time) + '</td>' +
            '  <td><b style="color:var(--input)">#' + esc(h.id) + '</b></td>' +
            '  <td style="color:var(--input)">' + num(h.input).toLocaleString() + '</td>' +
            '  <td style="color:var(--reuse)">' + num(h.kvReuse).toLocaleString() + '</td>' +
            '  <td>' + r.effHtml + '</td>' +
            '  <td>' + r.lostHtml + '</td>' +
            '  <td>' + r.statusHtml + '</td>' +
            '  <td>' + r.skewHtml + '</td>' +
            '  <td style="color:var(--prefill)">' + r.prefillHtml + '</td>' +
            '  <td style="color:var(--decode)">' + num(h.decode).toLocaleString() + decodeSpeedStr + '</td>' +
            '  <td style="color:var(--mtp);font-size:10px">' + (h.specAccepted != null ? num(h.specAccepted).toLocaleString() + '/' + num(h.specGenerated).toLocaleString() + ' (' + h.specAcceptance + '%) | ' + num(h.specMeanLen).toFixed(2) : '—') + '</td>' +
            '</tr>';
        }).join('');
      }
    }

    // The diagnostic cells of a line: EFF%, PREFIX GAP, STATUS, SKEW, the DIVERGE tooltip.
    // kvPending (the live line only): the KV line (cached/restored) has not
    // arrived yet — we show "…" instead of a false 0% / COLD.
    function histCells(h, prevH) {
          const input = num(h.input);
          const reuse = num(h.kvReuse);

          let effHtml, lostHtml, statusHtml;
          if (h.kvPending) {
            effHtml = '<span class="badge-lost-zero">…</span>';
            lostHtml = '<span class="badge-lost-zero">…</span>';
            statusHtml = '<span class="status-tag" style="color:var(--dim);border:1px solid var(--border)">…</span>';
          } else {
            // 1. EFF % — the cache hit rate
            const effPct = input > 0 ? (reuse / input * 100).toFixed(1) : '0.0';
            const effClass = effPct >= 98.0 ? 'badge-eff-high' : (effPct >= 85.0 ? 'badge-eff-med' : 'badge-eff-low');
            effHtml = '<span class="' + effClass + '">' + effPct + '%</span>';

            // 2. PREFIX GAP — the prefix break relative to the previous turn
            // (prev input + prev decode − reuse): the agent trimmed the context.
            // We count ONLY on a real prefix match (reuse > 0):
            // a separate/short request without a match — not a "loss", but "—"
            lostHtml = '<span class="badge-lost-zero">—</span>';
            if (prevH && num(prevH.input) > 0 && reuse > 0) {
              const expected = num(prevH.input) + num(prevH.decode);
              const lost = expected - reuse;
              if (lost > 2) {
                lostHtml = '<span class="badge-lost-warn" title="' + t('tt_lost_expected', { n: expected.toLocaleString() }) + '">-' + lost.toLocaleString() + '</span>';
              } else if (lost >= 0) {
                lostHtml = '<span class="badge-lost-zero" style="color:var(--reuse)">✓ 0</span>';
              }
            }

            // 3. STATUS — the semantic cache status
            if (h.reprocessReason === 'FULL_REPROCESS' && num(h.lcp) > 0) {
              statusHtml = '<span class="status-tag status-cold" style="color:#f59e0b;border-color:#f59e0b" title="' + t('st_reproc') + '">REPROC</span>';
            } else if (reuse <= 0 || h.slotSource === 'empty') {
              statusHtml = '<span class="status-tag status-cold">COLD</span>';
            } else if (h.checkpointRestore) {
              statusHtml = h.slotSource === 'hot-vram'
                ? '<span class="status-tag status-rewind" title="' + t('st_hotcp') + '">HOT+CP ↺</span>'
                : '<span class="status-tag status-ram" title="' + t('st_ramvram') + '">RAM→VRAM</span>';
            } else if (h.slotSource === 'hot-vram') {
              statusHtml = '<span class="status-tag status-hot" title="' + t('st_hot') + '">HOT VRAM</span>';
            } else if (h.slotSource === 'prompt-cache-ram') {
              statusHtml = '<span class="status-tag status-ram" title="' + t('st_ram') + '">RAM CACHE</span>';
            } else {
              statusHtml = '<span class="status-tag status-ram" title="' + t('st_cache') + '">CACHE</span>';
            }
          }

          // 4. SKEW — the overpay for the checkpoint step (LCP − restored)
          const skew = num(h.cpSkew);
          const skewHtml = skew > 0
            ? '<span class="skew-tag" title="' + t('st_skew', { lcp: num(h.lcp).toLocaleString(), reuse: reuse.toLocaleString() }) + '">-' + skew.toLocaleString() + '</span>'
            : '<span style="color:var(--dim)">0</span>';

          // 5. PREFILL — tokens + the measured speed (dim 9.5px);
          // prefill = 0 (a full cache hit) — just "0" without the speed
          const prefill = num(h.prefill);
          const prefillSpeed = num(h.prefillSpeed, 0);
          const prefillHtml = prefill > 0
            ? (h.prefillSource === 'estimated' ? '~' : '') + prefill.toLocaleString()
              + (prefillSpeed > 0 ? ' <span style="font-size:9.5px;color:var(--dim)">(' + prefillSpeed.toFixed(0) + ' t/s)</span>' : '')
            : '0';

          // 6. The DIVERGE tooltip — the exact token divergence point
          let titleAttr = '';
          if (h.divergeInfo) {
// the backslashes are doubled: the code is inside the HTML_PAGE template literal (otherwise it breaks <script>)
            titleAttr = ' title="' + t('desync_title', { pos: h.divergeInfo.pos }) + '\\n' + t('desync_slot') + ': [' + esc(h.divergeInfo.tokA) + '] \\\'' + esc(h.divergeInfo.pieceA) + '\\\'\\n' + t('desync_req') + ': [' + esc(h.divergeInfo.tokB) + '] \\\'' + esc(h.divergeInfo.pieceB) + '\\\'"';
          }

          return { effHtml, lostHtml, statusHtml, skewHtml, prefillHtml, titleAttr };
    }

    let sseWorking = false;
    let initialLoadDone = false;
    let latestState = null, renderQueued = false, clockOffset = 0, pollBusy = false;

    function scheduleRender(s) {
      latestState = s;
      // ⏸: a local pause blocks the render ONLY in LIVE (in PLAY/load-log
      // the pause is server-side — we do not block the render, the state comes from s.player)
      if (paused && isLiveMode()) return;
      if (renderQueued) return;
      renderQueued = true;
      requestAnimationFrame(() => {
        renderQueued = false;
        try { if (latestState) renderState(latestState); } catch (err) { console.error(err); }
      });
    }
    function onSnapshot(s) {
      if (s && s.serverNow) clockOffset = s.serverNow - Date.now();
      // C1: the cache by versions — empty arrays in the snapshot mean "no changes"
      if (s) {
        if (s.eventsRev !== clientEventsRev) { clientEventsRev = s.eventsRev; clientEvents = s.events || []; }
        if (s.historyRev !== clientHistoryRev) { clientHistoryRev = s.historyRev; clientHistory = s.history || []; }
        // A1: in non-LIVE the pause and the speed — the server state, there are no local copies
        if (s.mode !== 'LIVE' && s.player) {
          if (ctlPlay) ctlPlay.textContent = s.player.paused ? '▶' : '⏸';
          if (ctlSpeed && document.activeElement !== ctlSpeed) ctlSpeed.value = String(s.player.speed);
        }
      }
      scheduleRender(s);
    }
    setInterval(() => { if (latestState) scheduleRender(latestState); }, 1000);

    if (window.EventSource) {
      const es = new EventSource('/events');
      es.onopen = () => { sseWorking = true; initialLoadDone = true; };
      es.onmessage = (e) => {
        sseWorking = true;
        initialLoadDone = true;
        try { onSnapshot(JSON.parse(e.data)); } catch (_) {}
      };
      es.onerror = () => { sseWorking = false; };
    }

    async function pollFallback() {
      if (pollBusy || sseWorking) return;
      pollBusy = true;
      try {
        const res = await fetch('/api/state', { cache: 'no-store' });
        if (res.ok) {
          onSnapshot(await res.json());
          initialLoadDone = true;
        } else if (initialLoadDone && !sseWorking) {
          const b = document.getElementById('status-badge');
          if (b) {
            b.className = 'badge-mini-status badge-off';
            b.innerText = 'MONITOR OFFLINE';
          }
          const dot = document.getElementById('tier-dot');
          if (dot) dot.className = 'status-dot dot-offline';
          const ts = document.getElementById('tier-server-status');
          if (ts) ts.innerText = 'OFFLINE';
        }
      } catch (_) {
        if (initialLoadDone && !sseWorking) {
          const b = document.getElementById('status-badge');
          if (b) {
            b.className = 'badge-mini-status badge-off';
            b.innerText = 'MONITOR OFFLINE';
          }
          const dot = document.getElementById('tier-dot');
          if (dot) dot.className = 'status-dot dot-offline';
          const ts = document.getElementById('tier-server-status');
          if (ts) ts.innerText = 'OFFLINE';
        }
      } finally {
        pollBusy = false;
      }
    }
    // ── The footer toolbar: pause / again / speed / log / clear ──
    // A1: in LIVE (stdin) there is no player — the pause is local, the speed/again dim out,
    // in PLAY/load-log the commands go to the server (POST /api/player)
    let paused = false; // a local pause — LIVE only (in PLAY/load-log the pause is server-side)
    const isLiveMode = () => latestState && latestState.mode === 'LIVE';
    const ctlPlay = document.getElementById('ctl-play');
    if (ctlPlay) ctlPlay.addEventListener('click', () => {
      if (isLiveMode()) {
        paused = !paused;
        ctlPlay.textContent = paused ? '▶' : '⏸';
        if (!paused && latestState) scheduleRender(latestState);
      } else {
        // The button only sends the command; the state comes from s.player.paused (onSnapshot)
        const action = (latestState && latestState.player && latestState.player.paused) ? 'resume' : 'pause';
        fetch('/api/player', { method: 'POST', body: JSON.stringify({ action }) }).catch(() => {});
      }
    });
    const ctlRestart = document.getElementById('ctl-restart');
    if (ctlRestart) ctlRestart.addEventListener('click', () => {
      lastEventsHash = '';
      lastHistHash = '';
      lastLiveHash = '';
      document.getElementById('log-feed').innerHTML = '';
      document.getElementById('hist-rows').innerHTML = '';
      document.getElementById('hist-live-row').innerHTML = '';
      if (!isLiveMode()) {
        fetch('/api/player', { method: 'POST', body: JSON.stringify({ action: 'restart' }) }).catch(() => {});
      }
      if (latestState) scheduleRender(latestState);
    });
    const ctlLive = document.getElementById('ctl-live');
    if (ctlLive) ctlLive.addEventListener('click', () => {
      // PLAYBACK → LIVE: the server stops the player, resets the engine, returns to the stdin stream
      fetch('/api/player', { method: 'POST', body: JSON.stringify({ action: 'live' }) }).catch(() => {});
    });
    const ctlSpeed = document.getElementById('ctl-speed');
    if (ctlSpeed) {
      // The speed — the server state (s.player.speed, onSnapshot); we do not store it in localStorage
      ctlSpeed.addEventListener('change', () => {
        if (!isLiveMode()) {
          fetch('/api/player', { method: 'POST', body: JSON.stringify({ action: 'speed', speed: parseFloat(ctlSpeed.value) }) }).catch(() => {});
        }
      });
    }
    const ctlClear = document.getElementById('ctl-clear');
    if (ctlClear) ctlClear.addEventListener('click', () => {
      document.getElementById('log-feed').innerHTML = '';
      document.getElementById('hist-rows').innerHTML = '';
      document.getElementById('hist-live-row').innerHTML = '';
      lastEventsHash = '';
      lastHistHash = '';
      lastLiveHash = '';
      fetch('/api/clear', { method: 'POST' }).catch(() => {});
    });
    const ctlFile = document.getElementById('ctl-file');
    if (ctlFile) ctlFile.addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      // #10: the client limit — we do not read >2 MB into the browser memory (the server
      // will answer 413 anyway, but we prevent the tab OOM in advance)
      if (f.size > 2 * 1024 * 1024) {
        alert(t('alert_log_big', { mb: (f.size / 1048576).toFixed(1) }));
        e.target.value = '';
        return;
      }
      const r = new FileReader();
      r.onload = (evt) => {
        const text = String(evt.target.result || '');
        document.getElementById('log-feed').innerHTML = '';
        document.getElementById('hist-rows').innerHTML = '';
        document.getElementById('hist-live-row').innerHTML = '';
        lastEventsHash = '';
        lastHistHash = '';
        lastLiveHash = '';
        fetch('/api/load-log', { method: 'POST', body: text })
          .then(res => { if (!res.ok) alert(t('alert_load_fail_http', { status: res.status })); })
          .catch(() => { alert(t('alert_load_fail')); });
      };
      r.readAsText(f);
      e.target.value = '';
    });

    setInterval(pollFallback, 400);
    pollFallback();
  })();
  </script>
  <div id="flow-overlay" class="flow-overlay">
    <div class="flow-viewer-toolbar">
      <span class="flow-viewer-title">INFERENCE PIPELINE</span>
      <button id="fv-zoom-out" type="button" title="Zoom out">&minus;</button>
      <span id="fv-zoom-val">100%</span>
      <button id="fv-zoom-in" type="button" title="Zoom in">+</button>
      <button id="fv-reset" type="button" title="Fit to view">RESET</button>
      <button id="fv-toggle-info" type="button" data-i18n="metrics_guide_btn" style="border-color:var(--input); color:var(--input);">📖 METRICS &amp; FORMULAS</button>
      <button id="fv-close" type="button" title="Close (Esc)">&times;</button>
    </div>
    <div id="flow-viewer-canvas" class="flow-viewer-canvas"></div>
    <div id="pipeline-metrics-guide" class="metrics-guide-drawer">
      <div class="metrics-grid">

        <!-- 1. INPUT -->
        <div class="metric-card">
          <div class="metric-head" style="color:var(--input)">
            <span class="m-pill" style="border-color:var(--input)">INPUT</span>
            <b data-i18n="metric_input_title">Input context</b>
          </div>
          <p class="m-desc" data-i18n="metric_input_desc">Total volume of Token IDs received from the client (system prompt + message history + new question).</p>
          <code class="m-code" data-i18n="metric_input_code">prompt_tokens = task.n_tokens</code>
          <div class="m-why"><b data-i18n="why_label">Why:</b> <span data-i18n="metric_input_why">Baseline reference point. Shows how much data the model has to consume in total.</span></div>
        </div>

        <!-- 2. KV REUSE -->
        <div class="metric-card">
          <div class="metric-head" style="color:var(--reuse)">
            <span class="m-pill" style="border-color:var(--reuse)">KV REUSE</span>
            <b data-i18n="metric_kv_title">Cache hit</b>
          </div>
          <p class="m-desc" data-i18n="metric_kv_desc">Number of tokens whose keys/values (KV) already reside in VRAM or were restored from Host RAM checkpoints.</p>
          <code class="m-code" data-i18n="metric_kv_code">reused = cached_n_tokens || restored_checkpoint</code>
          <div class="m-why"><b data-i18n="why_label">Why:</b> <span data-i18n="metric_kv_why">The main accelerator. These tokens do not need to be recomputed on the GPU.</span></div>
        </div>

        <!-- 3. EFF% -->
        <div class="metric-card">
          <div class="metric-head" style="color:#22c55e">
            <span class="m-pill" style="border-color:#22c55e">EFF%</span>
            <b data-i18n="metric_eff_title">Cache Hit Rate</b>
          </div>
          <p class="m-desc" data-i18n="metric_eff_desc">Caching efficiency in percent. How "free" the request is for the GPU.</p>
          <code class="m-code" data-i18n="metric_eff_code">eff = (kv_reuse / input) * 100%</code>
          <div class="m-why"><b data-i18n="norm_label">Norm:</b> <span data-i18n="metric_eff_why">≥98% — instant dialog response; &lt;80% — cold start or topic change.</span></div>
        </div>

        <!-- 4. PREFIX GAP -->
        <div class="metric-card">
          <div class="metric-head" style="color:var(--red)">
            <span class="m-pill" style="border-color:var(--red)">PREFIX GAP</span>
            <b data-i18n="metric_gap_title">Prefix gap</b>
          </div>
          <p class="m-desc" data-i18n="metric_gap_desc">How many tokens from the previous turn's context missed the current request's cache.</p>
          <code class="m-code" data-i18n="metric_gap_code">gap = (prev_input + prev_decode) - reuse</code>
          <div class="m-why"><b data-i18n="why_label">Why:</b> <span data-i18n="metric_gap_why">Detects context truncation (agent frameworks, topic change). If &gt; 0 — part of the prefix was lost.</span></div>
        </div>

        <!-- 5. STATUS -->
        <div class="metric-card">
          <div class="metric-head" style="color:var(--kv-save)">
            <span class="m-pill" style="border-color:var(--kv-save)">STATUS</span>
            <b data-i18n="metric_status_title">Physical KV source</b>
          </div>
          <p class="m-desc" data-i18n="metric_status_desc">Where the cache tensors physically come from: directly from a hot GPU slot, from RAM, or from scratch.</p>
          <code class="m-code" data-i18n="metric_status_code">HOT VRAM | RAM→VRAM (checkpoint) | COLD</code>
          <div class="m-why"><b data-i18n="why_label">Why:</b> <span data-i18n="metric_status_why_pre">Shows how the</span> <code>--cache-ram</code> <span data-i18n="metric_status_why_post">mechanism and the PCIe bus work during save/load.</span></div>
        </div>

        <!-- 6. SKEW -->
        <div class="metric-card">
          <div class="metric-head" style="color:#f59e0b">
            <span class="m-pill" style="border-color:#f59e0b">SKEW</span>
            <b data-i18n="metric_skew_title">Quantization-step overpay</b>
          </div>
          <p class="m-desc" data-i18n="metric_skew_desc">Difference between the exact prefix match (LCP) and the position of the nearest checkpoint in RAM.</p>
          <code class="m-code" data-i18n="metric_skew_code">skew = lcp_tokens - kv_restored_tokens</code>
          <div class="m-why"><b data-i18n="why_label">Why:</b> <span data-i18n="metric_skew_why">Shows how many tokens had to be recomputed on the GPU solely because of the checkpoint save step (spacing).</span></div>
        </div>

        <!-- 7. PREFILL -->
        <div class="metric-card">
          <div class="metric-head" style="color:var(--prefill)">
            <span class="m-pill" style="border-color:var(--prefill)">PREFILL</span>
            <b data-i18n="metric_prefill_title">New GPU compute</b>
          </div>
          <p class="m-desc" data-i18n="metric_prefill_desc">New tokens actually computed on the tensor cores. Parallel matrix multiplication (GEMM).</p>
          <code class="m-code" data-i18n="metric_prefill_code">delta = input - kv_reuse (ms / tokens = t/s)</code>
          <div class="m-why"><b data-i18n="bottleneck_label">Bottleneck:</b> <span data-i18n="metric_prefill_why">GPU TFLOPS limit. The smaller the delta — the faster generation starts.</span></div>
        </div>

        <!-- 8. DECODE -->
        <div class="metric-card">
          <div class="metric-head" style="color:var(--decode)">
            <span class="m-pill" style="border-color:var(--decode)">DECODE</span>
            <b data-i18n="metric_decode_title">Answer generation</b>
          </div>
          <p class="m-desc" data-i18n="metric_decode_desc">Step-by-step token output, one after another. Matrix-vector multiplication (GEMV).</p>
          <code class="m-code" data-i18n="metric_decode_code">tokens_generated / eval_time_seconds = t/s</code>
          <div class="m-why"><b data-i18n="bottleneck_label">Bottleneck:</b> <span data-i18n="metric_decode_why">GPU memory bus bandwidth (GB/s). Every token requires reading all model weights.</span></div>
        </div>

        <!-- 9. SPEC (MTP + NGRAM) -->
        <div class="metric-card">
          <div class="metric-head" style="color:var(--mtp)">
            <span class="m-pill" style="border-color:var(--mtp)">SPEC (MTP+NGRAM)</span>
            <b data-i18n="metric_spec_title">Speculative draft</b>
          </div>
          <p class="m-desc" data-i18n="metric_spec_desc">Multi-Token Prediction / n-gram. A small network or heuristic predicts several tokens ahead. SPEC TOTAL — all implementations combined (per-task); DRAFT-MTP / NGRAM-MOD — session CUMULATIVE.</p>
          <code class="m-code" data-i18n="metric_spec_code">acceptance_rate (accepted/generated) + mean_len = 1 + Σ acc_rate_per_pos</code>
          <div class="m-why"><b data-i18n="why_label">Why:</b> <span data-i18n="metric_spec_why">Allows obtaining several tokens per one weight-read tick from VRAM, speeding up DECODE.</span></div>
        </div>

      </div>
    </div>
  </div>
  <script>
  (function () {
    var wrap = document.querySelector('.flow-wrap'); // may be absent (the footer was removed) — the trigger is then only LEGEND
    var overlay = document.getElementById('flow-overlay');
    var canvas = document.getElementById('flow-viewer-canvas');
    var zoomVal = document.getElementById('fv-zoom-val');
    if (!overlay || !canvas || !zoomVal) return;
    var svgEl = null, scale = 1, tx = 0, ty = 0, fitScale = 1;
    function apply() {
      if (!svgEl) return;
      svgEl.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
      zoomVal.textContent = Math.round((scale / fitScale) * 100) + '%';
    }
    function fit() {
      if (!svgEl) return;
      var vb = svgEl.viewBox && svgEl.viewBox.baseVal;
      var w = vb ? vb.width : svgEl.clientWidth;
      var h = vb ? vb.height : svgEl.clientHeight;
      var cw = canvas.clientWidth, ch = canvas.clientHeight;
      fitScale = Math.min(cw / w, ch / h) * 0.95;
      scale = fitScale;
      tx = (cw - w * scale) / 2;
      ty = (ch - h * scale) / 2;
      apply();
    }
    function open() {
      var src = document.querySelector('.flow-diagram svg');
      if (!src) {
        // Mermaid is not loaded (a LAN without jsdelivr) or a click before mermaid.run():
        // the metric formulas do not depend on the diagram — we open the window with a placeholder
        canvas.innerHTML = '<div style="color:var(--dim);padding:20px">' + window.__wwwT('mermaid_unavailable') + '</div>';
        overlay.classList.add('open');
        return;
      }
      canvas.innerHTML = '';
      svgEl = src.cloneNode(true);
      svgEl.removeAttribute('id');
      var vb = src.viewBox && src.viewBox.baseVal;
      if (vb) {
        svgEl.setAttribute('width', vb.width);
        svgEl.setAttribute('height', vb.height);
      }
      svgEl.removeAttribute('style');
      canvas.appendChild(svgEl);
      overlay.classList.add('open');
      fit();
    }
    function close() {
      overlay.classList.remove('open');
      canvas.innerHTML = '';
      svgEl = null;
    }
    function zoomAt(factor, cx, cy) {
      var ns = Math.max(fitScale * 0.2, Math.min(fitScale * 20, scale * factor));
      var k = ns / scale;
      tx = cx - (cx - tx) * k;
      ty = cy - (cy - ty) * k;
      scale = ns;
      apply();
    }
    if (wrap) wrap.addEventListener('click', open);
    document.getElementById('fv-close').addEventListener('click', close);
    document.getElementById('fv-reset').addEventListener('click', fit);
    document.getElementById('fv-zoom-in').addEventListener('click', function () {
      zoomAt(1.25, canvas.clientWidth / 2, canvas.clientHeight / 2);
    });
    document.getElementById('fv-zoom-out').addEventListener('click', function () {
      zoomAt(0.8, canvas.clientWidth / 2, canvas.clientHeight / 2);
    });
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault();
      var r = canvas.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    var drag = null;
    canvas.addEventListener('mousedown', function (e) {
      drag = { x: e.clientX, y: e.clientY, tx: tx, ty: ty, moved: 0 };
      canvas.classList.add('dragging');
    });
    window.addEventListener('mousemove', function (e) {
      if (!drag) return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.moved = Math.max(drag.moved, Math.abs(dx) + Math.abs(dy));
      tx = drag.tx + dx;
      ty = drag.ty + dy;
      apply();
    });
    window.addEventListener('mouseup', function (e) {
      if (!drag) return;
      var wasDrag = drag.moved > 4;
      drag = null;
      canvas.classList.remove('dragging');
      if (!wasDrag && e.target === canvas) close();
    });
    canvas.addEventListener('dblclick', fit);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && overlay.classList.contains('open')) close();
    });
    window.addEventListener('resize', function () {
      if (overlay.classList.contains('open')) fit();
    });
    var btnToggleInfo = document.getElementById('fv-toggle-info');
    var metricsDrawer = document.getElementById('pipeline-metrics-guide');
    if (btnToggleInfo && metricsDrawer) {
      btnToggleInfo.addEventListener('click', function () {
        metricsDrawer.classList.toggle('collapsed');
        fit();
      });
    }
    var helpLink = document.getElementById('open-metrics-help');
    if (helpLink) {
      helpLink.addEventListener('click', function () {
        open();
        if (metricsDrawer) metricsDrawer.classList.remove('collapsed');
      });
    }
  })();
  </script>
</body>
</html>
`;

// The same-origin check for POST: a foreign page in the browser can send a simple POST
// (text/plain, without a preflight) — the effect (reset/player) will execute, even if
// the browser blocks reading the response. Without an Origin (curl/CLI/browser stubs) — let through.
function sameOrigin(req) {
  const h = req.headers || {};
  if (!h.origin) return true;
  try { return new URL(h.origin).host === h.host; } catch (_) { return false; }
}

const server = http.createServer((req, res) => {
  if (req.url === '/' || req.url === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
    res.end(HTML_PAGE);
  } else if (req.url === '/favicon.ico') {
    res.writeHead(204);
    res.end();
  } else if (req.url === '/events') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      // #16: without Access-Control-Allow-Origin — the UI is same-origin, a wildcard is not needed
      'X-Accel-Buffering': 'no'
    });
    if (res.flushHeaders) res.flushHeaders();
    if (res.socket && typeof res.socket.setNoDelay === 'function') {
      res.socket.setNoDelay(true);
    }

    const heartbeat = setInterval(() => {
      try {
        res.write(': heartbeat\n\n');
        if (typeof res.flush === 'function') res.flush();
      } catch (_) {
        clearInterval(heartbeat);
        sseClients.delete(res);
      }
    }, 15000);

    res.write(`data: ${snapshotJson(true)}\n\n`); // C1: on connect — the full snapshot
    if (typeof res.flush === 'function') res.flush();
    res._evRev = state.eventsRev; // C1: per-client versions — deltas only afterwards
    res._histRev = state.historyRev;
    sseClients.add(res);

    req.on('close', () => {
      clearInterval(heartbeat);
      sseClients.delete(res);
    });
  } else if (req.url === '/api/state') {
    res.writeHead(200, {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store'
      // #16: without Access-Control-Allow-Origin — the UI is same-origin, a wildcard is not needed
    });
    res.end(snapshotJson(true)); // C1: the full snapshot
  } else if (req.url === '/api/logs') {
    // Safe reading of the last 64 KB of the log file (OOM protection for files of any size)
    // A4: in PLAY mode we serve the tail of the PLAYED file, not LOG_FILE
    const logsFile = PLAY !== -1 ? argv[PLAY + 1] : LOG_FILE;
    fs.stat(logsFile, (err, stats) => {
      if (err || stats.size === 0) {
        res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
        return res.end(`Log file ${logsFile} does not exist yet or is empty.\n`);
      }
      const READ_SIZE = Math.min(stats.size, 64 * 1024);
      const buffer = Buffer.alloc(READ_SIZE);
      fs.open(logsFile, 'r', (openErr, fd) => {
        if (openErr) {
          res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
          return res.end(`Failed to open log file: ${openErr.message}\n`);
        }
        fs.read(fd, buffer, 0, READ_SIZE, stats.size - READ_SIZE, (readErr) => {
          fs.close(fd, () => {});
          if (readErr) {
            res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
            return res.end(`Read error: ${readErr.message}\n`);
          }
          const text = buffer.toString('utf8');
          const lines = text.split('\n');
          // A4: slice(1) only if the file is larger than the read window (the first line was cut)
          const cleanLines = (stats.size > READ_SIZE ? lines.slice(1) : lines).filter(l => !l.includes('update_slots: all slots are idle'));
          const tail = cleanLines.slice(-150).join('\n');
          res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
          res.end(tail + '\n');
        });
      });
    });
  } else if (req.url === '/api/clear' && req.method === 'POST') {
    if (!sameOrigin(req)) { res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' }); return res.end('forbidden'); }
    // The "Clear" toolbar: reset of the event feed and the history (the task is not touched)
    state.events = [];
    state.history = [];
    state.eventsRev++; state.historyRev++; // B2/C1
    broadcastState();
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end('{"ok":true}');
  } else if (req.url === '/api/load-log' && req.method === 'POST') {
    if (!sameOrigin(req)) { res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' }); return res.end('forbidden'); }
    // The "Log" toolbar: A2 — a line-by-line run of a user .log through the player (A1).
    // Works in LIVE and PLAYBACK: switches state.mode to 'PLAYBACK'
    // (the live stdin lines are written to disk meanwhile, but not parsed)
    const MAX_LOG = 2 * 1024 * 1024; // 2 MB — OOM protection
    req.setEncoding('utf8');
    let body = '';
    let tooBig = false;
    req.on('data', (chunk) => {
      if (tooBig) return;
      body += chunk;
      if (Buffer.byteLength(body, 'utf8') > MAX_LOG) { tooBig = true; body = ''; }
    });
    req.on('end', () => {
      if (tooBig) {
        res.writeHead(413, { 'Content-Type': 'text/plain; charset=utf-8' });
        return res.end('Log too large (max 2 MB)\n');
      }
      const lines = body.split('\n').filter(Boolean);
      const hasTs = lines.some(l => {
        const m = l.match(/^\[([^\]]+)\] /);
        return m && !Number.isNaN(Date.parse(m[1]));
      });
      if (!hasTs) {
        res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
        return res.end('need a LLAMA_LOG_FILE-format log (lines "[ISO] ...")\n');
      }
      resetAll();
      state.mode = 'PLAYBACK';
      playerStart(lines);
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ ok: true, mode: state.mode }));
    });
    return;
  } else if (req.url === '/api/player' && req.method === 'POST') {
    if (!sameOrigin(req)) { res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' }); return res.end('forbidden'); }
    // A1: the player control (pause/resume/restart/speed/live) — PLAY / PLAYBACK
    if (state.mode === 'LIVE') {
      res.writeHead(409, { 'Content-Type': 'application/json; charset=utf-8' });
      return res.end('{"ok":false,"error":"LIVE mode"}');
    }
    const MAX_PLAYER = 16 * 1024; // the player commands are tiny — protection against junk
    let body = '';
    let tooBig = false;
    req.on('data', (c) => {
      if (tooBig) return;
      body += c;
      if (Buffer.byteLength(body, 'utf8') > MAX_PLAYER) { tooBig = true; body = ''; }
    });
    req.on('end', () => {
      if (tooBig) {
        res.writeHead(413, { 'Content-Type': 'application/json; charset=utf-8' });
        return res.end('{"ok":false,"error":"body too large"}');
      }
      let action = '', speed = null;
      try { const j = JSON.parse(body || '{}'); action = j.action; speed = j.speed; } catch (_) {}
      if (action === 'pause') playerPause();
      else if (action === 'resume') playerResume();
      else if (action === 'restart') playerRestart();
      else if (action === 'speed' && speed != null) playerSetSpeed(speed);
      else if (action === 'live') {
        if (!LIVE_MODE) {
          res.writeHead(409, { 'Content-Type': 'application/json; charset=utf-8' });
          return res.end('{"ok":false,"error":"no live stream"}');
        }
        goLive();
      }
      else {
        res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
        return res.end('{"ok":false,"error":"unknown action"}');
      }
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ ok: true, player: state.player }));
    });
    return;
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
  }
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\x1b[1m\x1b[31m❌ ERROR: Port ${WEB_PORT} is already in use!\x1b[0m`);
    console.error(`Kill the old process: \x1b[33mfuser -k ${WEB_PORT}/tcp\x1b[0m\n`);
  } else {
    console.error(`\x1b[31mServer error:\x1b[0m`, err);
  }
  process.exit(1);
});

if (RING_MODEL !== -1) {
  // Test mode: a state snapshot -> the rings model (JSON to stdout).
  const s = JSON.parse(fs.readFileSync(argv[RING_MODEL + 1], 'utf8'));
  process.stdout.write(JSON.stringify(buildRingModel(s)) + '\n');
  process.exit(0);
} else if (LIVE_MODE) {
  state.mode = 'LIVE';
  server.listen(WEB_PORT, MONITOR_HOST, () => {
    console.log(`\x1b[1m\x1b[32m🚀 HARDENED REALTIME PIPELINE MONITOR STARTED:\x1b[0m http://localhost:${WEB_PORT}`);
    console.log(`📡 Connecting to llama-server: ${SERVER_URL}`);
    console.log(`📝 Disk raw log enabled: ${LOG_FILE}\n`);
  });
} else if (REPLAY !== -1) {
  // Wall clock only: ts/time/stateAt — new Date() (the virtual clock does not
  // cover them). serverNow does not appear in replay (there are no clients, flushState
  // exits before snapshotJson), but it stays in the set in case of live calls.
  // lastRestored/evictedAt/_doneSince/_processingStart — Date.now() inside
  // handleLine → determined by the virtual clock and they stay in the snapshot,
  // so the golden-diff sees regressions in the restore/evict branches.
  const VOLATILE = new Set(['ts', 'time', 'stateAt', 'serverNow']);
  state.mode = 'REPLAY';
  // The virtual clock: Date.now() = the timestamp of the current log line,
  // otherwise the sweep of evicted slots (>2000 ms) depends on the real loop speed
  let replayTime = 0;
  for (const raw of fs.readFileSync(argv[REPLAY + 1], 'utf8').split('\n')) {
    const m = raw.match(/^\[([^\]]+)\] /);
    if (m) {
      const t = Date.parse(m[1]);
      if (!Number.isNaN(t)) replayTime = t;
    }
    Date.now = () => replayTime;
    feedLogLine(raw); // A2: the single entry point (currentLogTs + handleLine)
  }
  Date.now = realNow;
  setTimeout(() => {
    state.events.forEach(e => delete e.id);
    fs.writeFileSync(argv[REPLAY + 2], JSON.stringify(state, (k, v) => VOLATILE.has(k) ? undefined : v, 2));
    process.exit(0);
  }, 2000);
} else {
  // PLAYBACK (CLI --play): the lines are fed by timestamps (time-lapse), without
  // writing to the log and without polling /slots. The server keeps working after the end
  // of the log — to inspect the final state in the browser. The mode — the same
  // 'PLAYBACK' as /api/load-log: the front knows only LIVE/PLAYBACK/REPLAY
  // (a separate 'PLAY' showed a green ONLINE in the header instead of PLAYBACK).
  state.mode = 'PLAYBACK';
  state.serverOnline = true; // /slots is not polled — the OFFLINE badge is not shown
  server.listen(WEB_PORT, MONITOR_HOST, () => {
    console.log(`\x1b[1m\x1b[32m🎬 PLAY MODE:\x1b[0m http://localhost:${WEB_PORT} (log: ${argv[PLAY + 1]}, speed x${PLAY_SPEED})`);
  });
  const lines = fs.readFileSync(argv[PLAY + 1], 'utf8').split('\n');
  // B5: without a single line with a valid [ISO] timestamp the playback
  // is impossible — an error before the server start
  const hasTs = lines.some(l => {
    const m = l.match(/^\[([^\]]+)\] /);
    return m && !Number.isNaN(Date.parse(m[1]));
  });
  if (!hasTs) {
    console.error('need a LLAMA_LOG_FILE-format log (lines "[ISO] ...")');
    process.exit(1);
  }
  playerStart(lines); // A1: the player (N4: the clock runs between the lines too)
}
