# Web-vs-API Response Comparison — Plan (Phase 1: ChatGPT)

## Context / Goal
Compare AI **product web responses** (what OneGlanse already captures from the real
ChatGPT/Gemini/etc. UI via Camoufox) against **raw API responses** for the *same prompt*, to see
**how the answers differ**.

The manager explicitly does **not** want the differences judged by another AI — OneGlanse's GEO
scoring is an LLM-judge, and that is being set aside for now. So **Phase 1 is a pure response-diff
study using transparent, deterministic metrics**. The team's own scoring logic comes later (Phase 2).

We start with **ChatGPT web vs ChatGPT API**, the provider already wired up in this repo.

---

## The three response sources (important framing)
The OpenAI API can run in two modes, which — together with the web product — give a clean
decomposition:

| Source | What it is |
|---|---|
| **api-raw** | Plain API completion. Model answers from training memory only — **no browsing, ~0 citations**. |
| **api-grounded** | API with the built-in **`web_search` tool** on. The model autonomously forms its own search queries, OpenAI runs them, and the model answers **with citation URLs**. This is the raw browsing capability, *unwrapped*. |
| **web** | ChatGPT the product = that same browsing **plus** ChatGPT's hidden system prompt, memory, source-card formatting, and product post-processing. |

This yields:
- **grounded − raw** = what *browsing* adds
- **web − grounded** = what the *ChatGPT product layer* adds on top of browsing
- **web − raw** = the total gap

Note: `web_search` is close to, but **not identical to**, ChatGPT web browsing (the product wraps it
in extra layers).

---

## Methodology

**Prompt set:** start with **10–15 prompts**, stratified by intent — branded ("is Trendyol good"),
comparison ("Trendyol vs Hepsiburada"), "best X for Y" ("best e-commerce site in Turkey"),
informational. Same language as the target market. Scale to **50–100** once the pipeline is trusted.

**Samples:** run each source **3 times per prompt**. LLM answers are non-deterministic, so multiple
samples let us separate a *real web-vs-API difference* from *run-to-run randomness*. Each web sample
uses a **fresh chat** (reset between runs) so samples are independent.

**Metrics — deterministic only (no AI judge):**

| Dimension | How |
|---|---|
| Length | chars / words |
| Citations | count + set of domains; **Jaccard overlap** of domains between sources |
| Brand mentions | string-match against a supplied brand list; which appear + **first-mention order/rank** |
| Structure | numbered list? headings? |
| Text similarity | lexical Jaccard on tokens + **embedding cosine similarity** (`text-embedding-3-small`) — a math distance, not an AI judging content |

Subjective dimensions (sentiment, "is it a recommendation") are **deferred to Phase 2 / the team's
own scoring** — not guessed by an LLM here.

**Report output:** per-prompt side-by-side (web | api-raw | api-grounded columns for each metric)
plus aggregates (mean length by source, mean citation count — expect web ≫ raw≈0, avg domain
overlap, % of prompts where the brand set differs, avg embedding similarity, within-source variance).

**Caveats to state in the report:**
- **`OPENAI_API_KEY` required** (currently empty in `.env`) — for API capture + embeddings.
- **Model matching:** ChatGPT web is a *system* (browsing + hidden system prompt + memory + tools),
  not just a model. Use the closest API model, keep it configurable, and state explicitly that the
  comparison is **products, not just models**.

---

## Implementation (standalone harness — no schema/UI/scoring changes)

### Reuse (do not rebuild)
- `apps/agent/src/run-test.ts` — template for a provider harness.
- `createAgent(provider)` → `{ page, cleanup }` — `apps/agent/src/core/createAgent.ts` (uses the
  stored Camoufox auth session; ChatGPT is already authed).
- `executePrompt(page, prompt, provider)` → `{ response, sources }` —
  `apps/agent/src/core/prompt-runner/executePrompt.ts`.
- `resetChatgptPage(page)` → fresh chat between samples —
  `apps/agent/src/core/providers/chatgpt/lib/pageLifecycle.ts`.
- `Source` type — `packages/types/src/types/sources.ts`.
- `openai` SDK — already a dependency of `packages/services` (no new install; installs are blocked in
  this env anyway). Used for API capture + embeddings via the existing `chatgpt` client in
  `packages/services/src/llm/index.ts`.

### New files
1. **`packages/services/src/apiCapture/openai.ts`** — `captureOpenAiResponse({ prompt, model, grounded })`
   → `{ response, sources }`. raw = `chatgpt.responses.create({ model, input })`; grounded = same +
   `tools: [{ type: "web_search" }]`, extracting `url_citation` annotations into `Source[]`.
2. **`packages/services/src/compare/diff.ts`** — pure deterministic functions: `computeMetrics`,
   `jaccard`, `lexicalSimilarity`, `embeddingSimilarity`, domain helpers.
3. **`packages/services/src/compare/report.ts`** — `buildComparisonReport(results, opts)` →
   `{ markdown, json, csv }`.
4. **`packages/services/src/compare/types.ts`** — `CapturedResponse`, `PromptResult`, `ReportOptions`.
5. **`apps/agent/src/run-compare.ts`** — entry script (mirrors `run-test.ts`): read config, capture
   web ×N + API raw/grounded ×N, run diff, write report under `apps/agent/compare-output/`.
6. **`apps/agent/compare.config.json`** — example input:
   `{ prompts: string[], brands?: string[], samples?: number (default 3), model?: string,
   modes?: ("raw"|"grounded")[] (default both), embeddings?: boolean }`.
7. Add a `"compare"` script to `apps/agent/package.json`.

### Explicitly OUT of Phase 1
- No ClickHouse `response_source` column, no schema migration, no UI, no `runAnalysis`/GEO scoring.
- ChatGPT only. Designed so Gemini/Claude/Perplexity API capture can be added later as sibling
  `apiCapture/*.ts` modules feeding the same diff/report code.

---

## Verification
1. Add a small `compare.config.json` (2–3 prompts, `samples: 1` for a fast smoke run).
2. Set `OPENAI_API_KEY` in `.env` (loaded the same way as the dev launcher that sources `.env`).
3. Run `pnpm --filter @oneglanse/agent exec node --loader ts-node/esm src/run-compare.ts`.
4. Confirm: web answer captured via Camoufox; API raw captured (no citations); API grounded
   captured (with citations); a report file written under `compare-output/`.
5. Spot-check the markdown report shows a readable side-by-side + aggregate section.
6. `pnpm --filter @oneglanse/services typecheck` passes for the new modules.
