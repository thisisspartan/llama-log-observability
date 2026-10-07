<img width="1280" height="698" alt="demo" src="https://github.com/user-attachments/assets/2d8dba04-e016-4348-bf76-c8a12b38b02b" />

# llama.cpp Log Monitor

Realtime dashboard for `llama-server` (llama.cpp): it parses the server log and
renders a live picture of the KV cache, context checkpoints, MTP / n-gram
speculation and request history.

**One file `www.js`. Pure Node.js. Zero dependencies.**

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![dependencies: none](https://img.shields.io/badge/dependencies-none-green.svg)
![runtime: Node stdlib](https://img.shields.io/badge/Node.js-stdlib-orange.svg)

## 🚀 Try it instantly (zero installation)

You don't need to clone this repo or run Node.js to inspect your logs:

👉 **[Launch the live web cockpit](https://thisisspartan.github.io/thisisspartan/)**

1. Open the link above.
2. Click **`📁 YOUR FILE`** in the top bar and pick your `llama-server` `.log`.
3. Watch KV-cache reuse, host-RAM checkpoints and MTP / n-gram speculation parse
   and render live in your browser — **zero install; your log never leaves
   your machine.**

> **How it works, honestly:** the demo page fetches the viewer code from GitHub
> (`raw.githubusercontent.com`) and the star count from `api.github.com`. Your
> `.log` is read locally by the browser (`FileReader`) and is never uploaded.
> Self-hosted, the dashboard served by `www.js` additionally references mermaid
> from the jsDelivr CDN.

## Why this exists

`llama-server` (llama.cpp) already prints everything you need — KV reuse, prompt-cache hits,
checkpoint saves/evictions, speculative-draft acceptance — but as flat text.
This tool turns that log into a live instrument panel: you see *where the cached
prefix came from*, *how much of the context is resident in VRAM vs host RAM*,
*whether a request is GEMM-bound (prefill) or GEMV-bound (decode)*, and *how
well MTP / n-gram drafts are being accepted* — without adding any instrumentation
to the server. It reads the log and the existing `/slots` + `/metrics` endpoints;
it changes nothing about how llama.cpp runs.

## Key features

- **GPU VRAM KV-SLOT** — donut reactor: KV REUSE / PREFILL / DECODE / FREE,
  phase (IDLE/PREFILL/DECODE/MTP/DONE), speed, bottleneck (GEMM vs GEMV).
- **KV provenance** — source of the cached prefix per llama.cpp canon:
  `hot-vram` (LCP slot match) / `prompt-cache-ram` (found better prompt) /
  `empty`; restored context checkpoint (Host RAM → VRAM) marked separately.
- **Host RAM checkpoints** — rack slots (identity by `n_tokens`), estimated
  occupied MiB, `cache state:` snapshots, red flash on eviction.
- **VRAM tape** — horizontal 0→n_ctx scale (reuse/prefill/decode/free) with
  projection of active RAM checkpoints onto the token scale.
- **MTP** — draft acceptance, tok/step, per-position accuracy.
- **NGRAM-MOD** — cumulative counters (acc/gen tokens, meanLen, posAcc).
- **REQUEST HISTORY** — rows per finished/cancelled task: KV reuse (VRAM/RAM),
  prefill (measured/estimated), decode, speed, MTP, LCP divergence
  (DIVERGE tooltip).
- **EVENT LOG** — live event stream with filters.
- **Server metrics** — n_ctx cross-check (`n_ctx_slot` line from the log vs
  `/slots` LIVE), queue, context overflow.

## Tested on

Everything in the demo log and the screenshots comes from one real production setup.

**Software**

- **OS:** Linux x86_64
- **llama-server (llama.cpp):** `0.5.0-dev` (build 11338, commit `dcd387a41`), built with GNU 14.3.0 for Linux x86_64

**Hardware**

- **GPU:** NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM — full offload (`-ngl 99`)
- **System RAM:** 64 GB DDR4 (host-RAM KV/checkpoint cache uses up to `--cache-ram 35840` ≈ 35 GiB)
- **CPU:** AMD Ryzen 7 5700X — server/monitor pinned to `--threads 8`

A 124k context on 16 GB of VRAM fits only because the KV cache is 4-bit
(`q4_0`) and the long tail of the context is checkpointed out to host RAM —
which is exactly the mechanism the dashboard visualizes.

**Model & runtime**

- **Model:** Qwen3.8-27B MTP GGUF (IQ3_S / GSQ quant)
- **Context:** `n_ctx = 124000`, KV cache `q4_0` (K and V)
- **Context checkpoints:** host-RAM checkpoints enabled (`--ctx-checkpoints 48`)
- **Speculative decoding:** `draft-mtp` + `ngram-mod`, draft length ≤ 5
- **Slots:** `--parallel 1` — a single slot, **required** (see below)

**Exact launch command**

```bash
stdbuf -oL -eL llama-server \
  -m /path/to/model.gguf \
  --alias Qwen3.8-27B-MTP \
  -ngl 99 --threads 8 --threads-batch 8 \
  -fa on --parallel 1 \
  -c 124000 -b 4480 -ub 224 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ctx-checkpoints 48 --checkpoint-min-step 1312 --cache-ram 35840 --no-cache-idle-slots \
  --seed 42 \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 5 --spec-draft-p-min 0.75 \
  --spec-ngram-mod-n-match 16 --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 36 \
  --temp 0.0 --top-k 1 --top-p 1.0 --min-p 0.0 \
  --repeat-penalty 1.0 --frequency-penalty 0.0 --presence-penalty 0.0 \
  --reasoning on \
  --jinja --chat-template-file /path/to/chat_template.jinja \
  --chat-template-kwargs '{"reasoning_effort":"medium","preserve_thinking":true,"max_tool_response_chars":20000}' \
  --metrics --host 0.0.0.0 --port 8080 -lv 4 2>&1 | MONITOR_HOST=0.0.0.0 node www.js
```

What each part does:

- `-ngl 99` — offload all layers to GPU; `--threads 8` — CPU thread budget.
- `-fa on` — flash-attention; `--parallel 1` — a single inference slot (**required**).
- `-c 124000` — 124k context; `-b 4480 -ub 224` — batch / ubatch for prefill.
- `--cache-type-k/v q4_0` — 4-bit KV cache (this is what the VRAM tape measures).
- `--ctx-checkpoints 48 --checkpoint-min-step 1312 --cache-ram 35840` — host-RAM
  context checkpoints (the "Host RAM checkpoints" rack in the dashboard).
- `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 5` — MTP + n-gram
  speculation (the MTP / NGRAM-MOD panels).
- `--metrics` — expose `/metrics` so the monitor can cross-check `n_ctx` and queue
  (**required** — the Server metrics panel reads it).
- `-lv 4` — verbose logging (**required** — the monitor parses these log lines); the pipe `| node www.js` is the whole integration:
  the monitor reads `llama-server` (llama.cpp) stdout, no server-side changes.

**Three obligatory flags: `-lv 4`, `--parallel 1`, `--metrics`.** These are the
only flags the monitor truly requires. `-lv 4` (verbose logging) is its entire
input — at lower verbosity the KV-reuse, checkpoint and speculation lines are
simply not printed, and the dashboard has nothing to render. `--metrics`
exposes `/metrics`, which the Server metrics panel reads to cross-check
`n_ctx` and queue. `--parallel 1` pins the server to a single inference slot.

**Why a single slot.** This is an educational monitor of *one* inference
stream: every KV block, phase transition and speculative draft belongs to a
single request, so the dashboard teaches what prefill / decode / MTP actually
do inside llama.cpp instead of interleaving several conversations into noise.
Run it with `--parallel 1` and follow one request end to end.

## Limitations

- **Log format.** The monitor parses `llama-server` output at `-lv 4`. The
  in-browser log upload (`/api/load-log`) expects lines prefixed with
  `[ISO timestamp]` — the format `www.js` itself writes to `LLAMA_LOG_FILE` —
  and a 2 MB size limit. Plain `llama-server` logs without timestamps are
  accepted in the live demo: it synthesizes timestamps for them client-side.
- **DIVERGE needs a patched build.** The exact token-divergence point
  (`[LCP-DEBUG] … <== DIVERGE` service lines) is not emitted by stock
  llama.cpp — it requires the server build used in “Tested on”. Everything
  else — KV provenance (`selected slot by LCP similarity`, prompt-cache
  `lcp =`), checkpoint save/evict, `statistics draft-mtp` — parses ordinary
  `llama-server` log output.
- **One task at a time.** With `--parallel > 1` the monitor follows a single
  active task and discards foreign-task lines — by design (see “Why a single
  slot”).
- **Node ≥ 18.** `www.js` uses the native global `fetch`.
- **No TTFT metric.** The monitor reports measured prefill/decode speed; it
  does not compute time-to-first-token.

## License

MIT — see [LICENSE](LICENSE).
