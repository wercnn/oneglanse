# HANDOFF — Turkcell Web-vs-API Study build

Resume doc for a fresh session. **No prior context needed beyond this file + the spec.**

- **Spec:** [turkcell-web-vs-api-study-spec.md](turkcell-web-vs-api-study-spec.md) (repo root).
- **Approved plan (same content):** `~/.claude/plans/synthetic-sleeping-hopper.md`.
- **Branch:** `local-run-fix-and-compare-plan` (Phase-1 harness already built + committed + pushed).
- **Status: CODE COMMITTED (2026-07-06) on this branch — 4 commits (steps 1–7). Corpus
  (`turkcell-mobile.json`) held back, still untracked, pending user prompt refinement.** All four
  post-build verification checks passed (grounded timer, brand resolver, CSV round-trip, per-prompt
  calibration). Not pushed. Pilot + two real study runs remain user-triggered.

## What this is
Additive extension of the existing web-vs-API comparison harness to run a **Turkcell brand-visibility PoC**
that answers: *is the web (Camoufox) capture leg worth its cost/fragility vs the Azure API?* Three deltas:
citation, share-of-voice, cost/reliability.

## Scope discipline (hard rules)
- **Additive only.** Do NOT touch `executePrompt` / `createAgent` / `resetChatgptPage` / production capture.
- Do NOT build RFP-platform items (dashboards, alerting, other engines, integrations, sentiment/hallucination).
- Run everything via compiled `dist` — `ts-node/esm` is broken on Node 24 (`ERR_REQUIRE_CYCLE_MODULE`).
- Provider is **Azure OpenAI gpt-5.5**; embeddings `text-embedding-3-small`, **api-version=preview only**.

## Three committed decisions
1. **logged_out persist-on-stop:** on `classifyError(err) === "logged_out"` in the web leg, stop the web loop
   but **break out to the normal end-of-run write** (do NOT return/throw past it) so partial-web + full-API
   still persists. API leg is independent of the ChatGPT session, so it still runs.
2. **Per-sample timing location:** add `durationMs?`/`attempts?`/`errorType?` to `CapturedResponse`, populated
   for **all three sources**. Web-only extras (recycle count, phase) stay in the existing `webSamples`.
   For grounded, time the **full round-trip** (the `await captureOpenAiResponse(...)` already returns after
   tool/search completes — just wrap timing around it).
3. **Light stemming:** brand matching is driven by the **explicit alias list** (folded); suffix-stemming is a
   light backstop only (over-stripping risks false brand collisions). Keyphrase stemming is light + flagged
   approximate.

## Environment / data facts
- E-commerce run JSON for no-cost checkpoints: `apps/agent/compare-output/report-2026-07-03T12-52-59-892Z.json`
  — **confirmed present**, has `results` (3 prompts) + `webSamples`. (No intent tags → `unspecified` on
  --analyze; fine for smoke.)
- `compare-output/` is gitignored.
- Existing reusable pieces: `computeMetrics`, `cosine`, `embedTexts`, `meanPairwiseCosine`, `meanCrossCosine`,
  `tokenize`, `hashText` (all `compare/diff.ts`); `getDomain` (`@oneglanse/utils`); `classifyError`
  (`@oneglanse/errors`); `buildComparisonReport` (`compare/report.ts`); `captureOpenAiResponse`
  (`apiCapture/openai.ts`); `--recompute` branch in `run-compare.ts` (template for `--analyze`/`--export`).

## 8-step build order + status

| Step | File(s) | Status |
|---|---|---|
| 1 | `packages/services/src/compare/types.ts` | ☑ done |
| 2 | `packages/services/src/compare/brands.ts` (new) | ☑ (İ/I/ı/i fold via NFD + `\p{Mn}` strip + dotless-I→i) |
| 3 | `packages/services/src/compare/diff.ts` | ☑ |
| 4 | `packages/services/src/compare/export.ts` (new) | ☑ |
| 5 | `packages/services/src/compare/analysis.ts` (new) | ☑ |
| 6 | `packages/services/src/index.ts` | ☑ |
| 7 | `apps/agent/src/run-compare.ts` | ☑ |
| 8 | `apps/agent/corpora/turkcell-mobile.json` (new) | ☐ code committed; corpus pending prompt refinement (untracked) |

### Step 1 — types.ts (additive)
```ts
export type Intent = "bilgi_alma" | "karsilastirma" | "satin_alma" | "marka_arama" | "unspecified";
export type BrandRole = "target" | "competitor" | "family" | "control";
export interface BrandEntry { canonical: string; aliases: string[]; role: BrandRole; domains?: string[]; }
// CapturedResponse += durationMs?: number; attempts?: number; errorType?: string;
// PromptResult    += intent?: Intent; brand?: string;
```

### Step 2 — brands.ts (new) — SoV correctness core
```ts
import type { BrandEntry } from "./types.js";
export function foldTr(s: string): string;                       // tr-lowercase + ç/ş/ğ/ı/ö/ü/İ→c/s/g/i/o/u/i
export function resolveBrandMentions(text: string, dict: BrandEntry[]): string[];   // canonical, first-mention order
export function resolveCitationBrands(domains: (string|null)[], dict: BrandEntry[]): string[]; // canonical via entry.domains
export function denominatorBrands(dict: BrandEntry[]): string[]; // canonicals with role target|competitor
```
Matching: fold text once; per entry, per alias, fold alias and find earliest boundary-anchored occurrence
(preceding char non-letter), **suffix-tolerant** (allow trailing letters/apostrophe, e.g. `Turkcell'in`).
Multiword aliases (`türk telekom`) matched as folded phrase. Return canonicals ordered by first index.
TT aliases all map to one `Türk Telekom` canonical.

**CORRECTNESS — Turkish İ/I/ı/i four-way (handle explicitly, deterministically):** plain
`.toLowerCase()` mishandles Turkish dotted/dotless I — capital `İ` can decompose to `i`+combining-dot, and
`I`→`i` is wrong for Turkish (should be `ı`). In `foldTr`, either **map `İ`/`I`/`ı`/`İ` explicitly BEFORE a
generic lowercase**, or use `toLocaleLowerCase('tr')` **and then** fold. The steps-1–3 checkpoint MUST
include a **capital-İ word** (e.g. `"İnternet"` or a brand containing `İ`), not only suffix cases.

### Step 3 — diff.ts additions (deterministic)
```ts
export const UNRELATED_CONTROL: string;                          // fixed off-topic Turkish sentence (floor anchor)
export const TURKISH_STOPWORDS: Set<string>;
export function extractKeyphrases(text: string, topN?: number): string[];        // content uni/bi-grams, stopword-filtered, light suffix-strip
export interface KeyphraseOverlap { jaccard: number; shared: string[]; uniqueA: string[]; uniqueB: string[]; }
export function keyphraseOverlap(a: string, b: string, topN?: number): KeyphraseOverlap;
// (calibration = analysis embeds responses + UNRELATED_CONTROL via embedTexts; floor = cosine vs control,
//  ceiling = meanPairwiseCosine within-source; cross positioned between. Reuses existing cosine/embedTexts.)
```

### Step 4 — export.ts (new)
```ts
import type { PromptResult } from "./types.js";
export function buildTranscript(results: PromptResult[]): string;   // md: per prompt, web|raw|grounded full text + citations + metrics
export function buildResponsesCsv(results: PromptResult[]): string; // 1 row per prompt×intent×brand×source×sample, full text col
```
Reuses `computeMetrics`; pulls `intent`/`brand` from `PromptResult`.

### Step 5 — analysis.ts (new) — the deliverable
```ts
import type { PromptResult, ReportOptions, BrandEntry } from "./types.js";
export interface RunData { options: ReportOptions; results: PromptResult[]; webSamples?: WebSampleMetaLike[]; }
export interface EvalOptions { brandDict: BrandEntry[]; }
export interface EvaluationReport { markdown: string; json: string; csv: string; }
export async function buildEvaluationAnalysis(runs: RunData[], opts: EvalOptions): Promise<EvaluationReport>;
```
Per **source × (overall + per intent)**:
- **Citation delta:** mean citations; domains web-only/grounded-only/shared; domain Jaccard web↔grounded.
- **SoV delta:** text-mention table (`resolveBrandMentions` → mentionRate/meanFirstRank/shareOfVoice over
  `denominatorBrands`) AND a **separate** citation-domain table (`resolveCitationBrands`); web-vs-grounded
  ranking agreement. **NEVER sum the two SoV tables.**
- **Cost/Reliability delta:** per source mean `durationMs` (from `results`), failures (from `results.error`),
  retries/recycles/logged_out (web, from `webSamples`); **throughput projection** = 100 prompts × daily × 5
  engines wall-time from web mean durationMs.
- **Cross-run stability:** only when `runs.length ≥ 2` (browse-decision stability, brand-set stability,
  citation drift). **CORRECTNESS — join prompts across runs on prompt TEXT (or a stable prompt id), NOT
  array index.** Reordering the config must not misalign run1 vs run2; browse-decision stability and
  citation-drift are meaningless if the join is positional.
Reuses steps 2–4 + `computeMetrics` + `embedTexts`/`meanPairwiseCosine`.

### Step 6 — index.ts
Add `export * from "./compare/brands.js"; ./compare/export.js; ./compare/analysis.js";`.

### Step 7 — run-compare.ts wiring
- `ConfigSchema.prompts`: `z.union([z.string(), z.object({ text, intent, brand?, targetBrands? })])`;
  normalize to objects. Add `brandDict: BrandEntry[]` (optional; if absent, synthesize from legacy `brands`
  string[] as role `competitor`). **Derive `ReportOptions.brands` = `denominator/canonical` names** so
  `buildComparisonReport` keeps working.
- Thread `intent`/`brand` into each `PromptResult` (+ persisted JSON).
- **CORRECTNESS — two brand columns will differ, and that's expected:** the legacy `buildComparisonReport`
  brand counts are **naive substring matching** (`computeMetrics` → `brandsInOrder`, plain `.toLowerCase()`,
  no alias/canonical/İ handling) and may disagree with `resolveBrandMentions`. The **authoritative SoV lives
  in `analysis.ts`** (canonical, folded, alias-resolved). Note this so nobody files a phantom bug when the
  report's brand column and the analysis SoV differ.
- `captureWeb`/`captureApi`: loops use `prompt.text`; set `durationMs`/`attempts`/`errorType` on the pushed
  `CapturedResponse` for **all** sources (time around each `executePrompt` / `captureOpenAiResponse`).
- **logged_out persist-on-stop:** flag + break out of both web loops (not return/throw) → reach normal write.
- `--analyze <run.json> [more…]` and `--export <run.json>` modes (mirror `--recompute` branch); write
  `analysis-<stamp>.{md,json,csv}`, `transcript-<stamp>.md`, `responses-<stamp>.csv`. Also write
  transcript+CSV on every normal run.

### Step 8 — corpora/turkcell-mobile.json
Brand dict (Turkcell target; Vodafone + Türk Telekom[canonical, aliases türk telekom/turk telekom/TT Mobil/
TT/Avea] competitors; TÜRKSAT control; Superonline family) + 20 tagged Turkish prompts, 5 each across the 4
intents, `samples: 5`, `embeddings: true`. Prompt wording is a runnable first draft (user will refine); TT
alias list must be precise.

## Checkpoints (no cost — do not skip)
```bash
# after steps 1–3
pnpm --filter @oneglanse/services build
#   then node-sanity brand resolver on: "Turkcell'in kapsama alanı", "Türk Telekom'a geçtim",
#   "TT Mobil'de", "Vodafone'dan" → canonicals resolve, TT aliases collapse to one.
#   Also: UNRELATED_CONTROL vs a Turkish telecom sentence ~low; a paraphrase ~high.

# after step 7
pnpm --filter @oneglanse/agent build
node apps/agent/dist/run-compare.js --analyze apps/agent/compare-output/report-2026-07-03T12-52-59-892Z.json
node apps/agent/dist/run-compare.js --export  apps/agent/compare-output/report-2026-07-03T12-52-59-892Z.json
#   → three delta tables + transcript + full-text CSV render (unspecified intent is fine).
```
Pilot (3–4 prompts, samples:2, ~$1) and the two real study runs (browser + ~$12–18 + 2 days) are
**user-triggered**, not part of the build.

## Open questions
- **TÜRKSAT status** — confirm it's still pre-launch / near-zero before locking it as the negative control.
- **Prompt corpus wording** — starter set only; user will refine the 20 Turkish prompts.
- **Turkish suffix-stemming depth** — using a light heuristic; revisit if keyphrase output looks noisy.
