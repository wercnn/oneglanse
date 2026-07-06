# Study Spec — Web vs API Capture, Worth-It Evaluation (Turkcell)

> Materialized from the approved plan (`~/.claude/plans/synthetic-sleeping-hopper.md`) so the repo is
> self-contained. This is the authoritative spec for the build.

## The one question this study answers
**Is the web-capture leg (Camoufox/ChatGPT UI) worth its compute cost and fragility, versus the Azure API —
for measuring AI brand visibility?** Internal PoC that de-risks the measurement engine and tells us which
capture method the eventual (Turkcell RFP) platform should rely on. *Not* a build of the RFP platform.

Decision rests on three deltas, web vs api-grounded (raw = no-retrieval baseline):
1. **Citation delta** — does web surface citations/domains the API doesn't?
2. **Share-of-Voice delta** — does the ranked competitor picture differ between web and API?
3. **Cost/Reliability delta** — can each method sustain the cadence a real platform needs?

## Study design (locked)
- **Target:** Turkcell (mobile / turkcell.com.tr). Corpus brand-tagged so Pasaj / Superonline slot in later.
- **Corpus:** 20 Turkish prompts, 5 per intent, RFP labels: `bilgi_alma`, `karsilastirma`, `satin_alma`,
  `marka_arama`.
- **Samples:** 5/source/prompt. **Runs:** 2, a day apart → `--analyze run1 run2`.
- **Sources:** web, api-raw, api-grounded (Azure gpt-5.5). **Cost/time:** ~$12–18, ~5 h over two days.

## Brand dictionary (the SoV denominator)
- **Target:** Turkcell — `turkcell`, suffix forms, domain `turkcell.com.tr`.
- **Competitors (denominator):** Vodafone (`vodafone`, `Vodafone Türkiye`, `vodafone.com.tr`);
  **Türk Telekom — ONE canonical** mapping `türk telekom`/`turk telekom`/`TT Mobil`/`TT`/`Avea` → single
  entity (most important matching rule).
- **Negative control:** `TÜRKSAT` (expect ~0; confirm current status).
- **Family (never competitor):** **Superonline = Turkcell's fiber brand** → tag Turkcell-family.
- Matching: `toLocaleLowerCase('tr')`, suffix-tolerant, ASCII/diacritic-folded. **Brand text-mention** and
  **brand-as-citation-domain** are SEPARATE signals (never summed).

## Engineering deliverables (additive only)
No changes to `executePrompt`/`createAgent`/`resetChatgptPage`/production capture. Run via compiled `dist`.

- **A. Intent + brand tags** — `run-compare.ts` config prompts `{text,intent,brand,targetBrands?}`; thread
  into `PromptResult` + JSON. `types.ts`: `Intent` + optional `intent?`/`brand?`.
- **B. Brand entity resolution** (`compare/brands.ts`) — `foldTr`, `resolveBrandMentions` (aliases→canonical,
  first-mention order, suffix-tolerant), `resolveCitationBrands`, role→denominator. Config `brandDict`.
- **C. Tier-1 semantics** (`diff.ts`) — calibrated cosine (`UNRELATED_CONTROL` floor / within-source ceiling);
  keyphrase shared/unique term lists (Turkish stopwords + light suffix-stem). Tier-2 sentence-coverage deferred.
- **D. Decision-delta analysis** (`compare/analysis.ts`) — `buildEvaluationAnalysis(runs, opts)`: citation
  delta, SoV delta (text-mention + citation-domain as SEPARATE tables; web-vs-grounded ranking agreement),
  cost/reliability delta + throughput projection, cross-run stability (≥2 runs). `--analyze` CLI mode.
- **E. Cost/reliability instrumentation + logged_out persist-on-stop** — per-call `durationMs` on all three
  sources; on `logged_out` stop the web leg but reach the normal end-of-run write (partial-web + full-API).
- **F. Exports** (`compare/export.ts`) — `buildTranscript` + `buildResponsesCsv` (full text); write each run;
  `--export` mode.
- **G. Data** — `apps/agent/corpora/turkcell-mobile.json` (brand dict + 20 tagged prompts).

### Out of scope
API parallelization; Tier-2 semantics; sentiment/hallucination (needs LLM-judge → Phase-2, flag to manager);
other engines; dashboards/GA4/Looker/Ahrefs/crawl/SSO/KVKK — all RFP-platform, not this study.

## RFP mapping (for the manager)
Scaled-down dress rehearsal of the RFP PoC (100 prompts, ≥5 competitors, per-brand, citation analysis,
visibility benchmark, exec summary): same intent taxonomy, same SoV + Citation-Share metrics, same
response-archive requirement — at 20×1 — answering the prerequisite the RFP is silent on: browser leg vs API.
Two RFP items outside the deterministic constraint (sentiment/brand-perception, hallucination detection) need
an LLM-judge layer excluded from Phase 1 — a Phase-2 decision that interacts with the web-vs-API choice.
