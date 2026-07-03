# Web-vs-API Response Comparison — Implemented (Phase 1: ChatGPT)

Companion to [web-vs-api-comparison-plan.md](web-vs-api-comparison-plan.md). That file is the
approved *plan*; this file records what was actually *built*, how it differs from the plan, how to
run it, and the findings from the first real runs.

Branch: `local-run-fix-and-compare-plan`. Status: **Phase 1 complete, committed, not pushed.**

---

## What it does

Compares, for the same prompt, three response sources and diffs them with **deterministic metrics
only (no LLM judge)**:

| Source | What it is |
|---|---|
| **web** | ChatGPT product via Camoufox (browsing + hidden system prompt + memory + product post-processing). |
| **api-raw** | Azure OpenAI completion, no tools — answers from training memory (~0 citations). |
| **api-grounded** | Same model + the built-in `web_search` tool — browses and answers with `url_citation` sources. |

Provider: **Azure OpenAI**, deployment **`gpt-5.5`** (not standard OpenAI). Embeddings via
**`text-embedding-3-small`** on the same Azure resource.

---

## Files (all new unless noted)

**`packages/services/`**
- `llm/azure.ts` — `azureOpenai()`, a lazy client on the Azure **v1 API surface**
  (`baseURL = <endpoint>/openai/v1/`, strips a trailing `/responses`, `defaultQuery {"api-version":"preview"}`,
  deployment name passed as `model`). The shared `chatgpt` client in `llm/index.ts` stays on standard
  OpenAI (runAnalysis untouched).
- `apiCapture/openai.ts` — `captureOpenAiResponse({ prompt, model, grounded })`. `grounded` adds
  `tools:[{type:"web_search"}]` and extracts `url_citation` annotations into `Source[]` (domain via
  `getDomain`). Same `{response, sources}` shape as the web harness.
- `compare/types.ts` — `CapturedResponse`, `PromptResult`, `ReportOptions`, `ResponseSource`, `ApiMode`.
- `compare/diff.ts` — deterministic metrics + similarity: `computeMetrics`, `jaccard`,
  `lexicalSimilarity`, `cosine`, `domainsOf`; embeddings: `embedTexts` (batched one-array-per-request,
  deduped/cached by `hashText` SHA-256), `cachedCosine`, `meanPairwiseCosine`, `meanCrossCosine`,
  `embeddingSimilarity`.
- `compare/report.ts` — `buildComparisonReport(results, opts) → {markdown, json, csv}`. Per-prompt
  side-by-side + aggregates + **embedding cosine section** (within-source baseline + cross-source pairs).
- `env.ts` (modified) — `AZURE_OPENAI_*` vars incl. `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
  (default `text-embedding-3-small`).
- `index.ts` (modified) — barrel exports for the new modules.

**`apps/agent/`**
- `src/run-compare.ts` — entry script. `captureWeb` (Camoufox, reliability layer), `captureApi`,
  report write, `--recompute` mode, capture persistence.
- `compare.config.json` — `{prompts, brands, samples, modes, embeddings}`. Turkish prompt set;
  `embeddings: true`.
- `package.json` (modified) — `"compare"` script (note: the script uses `ts-node/esm`, which is
  **broken on Node 24** — run the compiled `dist` instead, see below).

**Root**
- `.gitignore` (modified) — ignores `apps/agent/compare-output/`.

---

## How to run

`ts-node/esm` crashes under Node 24 (`ERR_REQUIRE_CYCLE_MODULE`), so **run the compiled JS**:

```bash
# build (services must be built before the agent — agent typechecks against its .d.ts)
pnpm --filter @oneglanse/services build
pnpm --filter @oneglanse/agent build

# full run (reads apps/agent/compare.config.json)
node apps/agent/dist/run-compare.js

# regenerate a report from a prior run's persisted captures, embeddings on, NO new captures
node apps/agent/dist/run-compare.js --recompute apps/agent/compare-output/report-<stamp>.json
```

Outputs: `apps/agent/compare-output/report-<stamp>.{md,json,csv}` (+ `recompute-<stamp>.*`, +
`failures/*.png`). The whole dir is gitignored. The JSON carries `results` (raw captures),
`webSamples` (per-web-sample diagnostics), and the computed report.

### `.env` (gitignored) — required keys
```
AZURE_OPENAI_ENDPOINT=https://<resource>.services.ai.azure.com/openai/v1/responses
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=gpt-5.5
AZURE_OPENAI_API_VERSION=preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```
Embeddings reuse the main endpoint/key; on this Azure v1 surface **only `api-version=preview` works**
for embeddings (dated GA versions are rejected).

---

## Metrics (deterministic — no AI judge)

- **Length**: chars, words.
- **Citations**: count, unique domains, domain **Jaccard** overlap between sources.
- **Brands**: string-match against a supplied list; which appear + **first-mention order**.
- **Structure**: numbered list? markdown headings?
- **Text similarity**: lexical **Jaccard** on tokens (representative sample per source) +
  **embedding cosine** (`text-embedding-3-small`).
- **Embedding baseline**: **within-source** = mean pairwise cosine among a source's own samples
  (the noise floor; `—` when <2 successful samples). **cross-source** = mean cosine over all sample
  pairs between two sources. Reading cross against the within-source floor makes it interpretable.

Subjective dimensions (sentiment, "is it a recommendation") remain **deferred to Phase 2** — not
guessed by an LLM.

---

## Reliability layer (web/Camoufox leg — "Tier 1")

The web capture is inherently flaky (live anti-bot SPA; the Camoufox session degrades after ~6
prompts). Production wraps every prompt in heavy retry machinery (`executePromptWithRetry` ×3 +
`runRetryCycle` ~30 attempts on fresh browsers + IP rotation). The harness reimplements a simplified
subset **in `run-compare.ts` only** (no changes to `executePrompt`/`createAgent`/`resetChatgptPage`):

- **Interleaved round-robin sampling** (round 1 of all prompts, then round 2…) so identical prompts
  aren't fired back-to-back.
- **Per-sample retry** (max 3), `resetChatgptPage` + exponential backoff **5s/15s/45s** between.
- **Browser recycle**: **proactively** every 5 captures, **reactively** before a 3rd attempt
  (navigation-reset alone doesn't recover a degraded session — a fresh browser does).
- **Recycle hardening**: recycle retries once after 15s; on double-failure the sample is recorded
  `recycle_failed`, skipped, and the batch **continues** (a dead browser never aborts the run).
- **2–6s jitter** between successful captures.
- **Per-sample metadata** in JSON `webSamples`: `attempts`, `recycledBrowser`, `durationMs`,
  `browsed`, and on failure `error`/`errorType`/`phase`. **Screenshot** on final failure.

Result: pre-fix run had **4/9** web samples fail; post-fix runs hit **0 failures**, recovering
transient `submission_failed` / navigation / `NS_BINDING_ABORTED` errors via retry + recycle.

Weaker than production: no IP rotation, no `logged_out` short-circuit, 3 attempts vs ~30. Fine for a
local, independent-sampling study; would need more for a proxied/VPS run.

---

## `--recompute` and persistence

Runs now persist raw `results` (captured response text + sources) in the report JSON.
`--recompute <run.json>` reloads them and rebuilds the report with embeddings **on**, doing **no new
captures** (only re-embeds identical text → identical numbers, verified byte-for-byte). Reports
written **before** this feature (commit `6fecff8`) have no `results` and cannot be backfilled.

---

## Key findings (from the real runs)

1. **Azure `gpt-5.5` supports the hosted `web_search` tool** — grounded mode works and returns real
   `url_citation` sources. (My earlier concern that Azure lacks it did not apply here.)
2. **Prompt language gates ChatGPT-web browsing.** Same query in **English** about Turkish brands →
   answered from memory, **0 citations**; in **Turkish** → **browsed, ~13 citations**. Use the target
   market's language (the plan already said so; the first config used English placeholders).
3. **Web browse decision is deterministic per prompt-type**: branded ("is X good") and comparison
   ("X vs Y") → memory, 0 citations; "best X for Y" → heavy browse (~13–15 citations, many domains).
   `api-raw` never cites; `api-grounded` always cites.
4. **Web and api-grounded browse different sources** — domain Jaccard `0.00` even when both browse
   (web pulls news/blogs/academic; grounded pulls analytics/gov like similarweb, gemius, ticaret.gov.tr).
5. **Embedding cosine**: within-source floor ~**0.94**; cross-source ~**0.86–0.91**, consistently
   ordered **web↔raw closest, web↔grounded furthest**. The three sources say broadly the *same thing*
   in meaning; the real differences are **citations, sourcing, and length** (grounded ~1.7× longer),
   which the deterministic metrics capture.
6. **Replication (two Turkish runs)**: brand set + first-mention order and web browse-decisions are
   **rock-stable**; only grounded citation counts, some domain sets, and web-P3 domains drift
   (magnitude noise, no categorical flips).

---

## Out of scope (Phase 1) / next
- No ClickHouse `response_source` column, no schema migration, no UI, no `runAnalysis`/GEO scoring.
- ChatGPT only. Designed so Gemini/Claude/Perplexity API capture can be added as sibling
  `apiCapture/*.ts` modules feeding the same diff/report code.
- Phase 2 = the team's own (subjective) scoring, on top of these deterministic diffs.

---

## Commits (branch `local-run-fix-and-compare-plan`, newest last)
- `c6772c8` feat(services): add Azure OpenAI capture for web-vs-api comparison
- `575ea2e` feat(services): add deterministic comparison metrics and reporting
- `371e448` feat(agent): add web-vs-api comparison harness (ChatGPT)
- `4b504ab` chore: gitignore comparison harness output
- `6fecff8` feat(services,agent): embedding similarity with within-source baseline, --recompute mode, recycle hardening
