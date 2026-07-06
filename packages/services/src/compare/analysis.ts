import {
	denominatorBrands,
	resolveBrandMentions,
	resolveCitationBrands,
} from "./brands.js";
import {
	UNRELATED_CONTROL,
	cachedCosine,
	embedTexts,
	jaccard,
	meanCrossCosine,
	meanPairwiseCosine,
} from "./diff.js";
import type {
	BrandEntry,
	CapturedResponse,
	Intent,
	PromptResult,
	ReportOptions,
	ResponseSource,
} from "./types.js";

/** Subset of the run-compare per-web-sample diagnostics this analysis reads. */
export interface WebSampleMetaLike {
	prompt: string;
	ok: boolean;
	attempts: number;
	recycledBrowser: boolean;
	durationMs: number;
	browsed: boolean;
	error?: string;
	errorType?: string;
	phase?: string;
}

/** One captured run (the shape persisted by run-compare into report JSON). */
export interface RunData {
	options: ReportOptions;
	results: PromptResult[];
	webSamples?: WebSampleMetaLike[];
}

export interface EvalOptions {
	brandDict: BrandEntry[];
}

export interface EvaluationReport {
	markdown: string;
	json: string;
	csv: string;
}

const SOURCES: ResponseSource[] = ["web", "api-raw", "api-grounded"];

const mean = (xs: number[]): number =>
	xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;

const fmt = (n: number, d = 2): string => n.toFixed(d);
const pct = (r: number): string => `${(r * 100).toFixed(0)}%`;

function isSuccess(c: CapturedResponse): boolean {
	return !c.error && c.response.trim().length > 0;
}

/** All captures (across runs/prompts) for a source, optionally scoped to intent. */
function collect(
	runs: RunData[],
	source: ResponseSource,
	intent?: Intent,
): CapturedResponse[] {
	const out: CapturedResponse[] = [];
	for (const run of runs) {
		for (const result of run.results) {
			const tag = result.intent ?? "unspecified";
			if (intent && tag !== intent) continue;
			for (const c of result.captures) if (c.source === source) out.push(c);
		}
	}
	return out;
}

function domainsOfCapture(c: CapturedResponse): (string | null)[] {
	return c.sources.map((s) => s.domain);
}

// ---------------------------------------------------------------------------
// Citation delta
// ---------------------------------------------------------------------------

interface CitationStat {
	source: ResponseSource;
	successCount: number;
	meanCitations: number;
	domains: string[];
}

function citationStats(runs: RunData[], intent?: Intent): CitationStat[] {
	return SOURCES.map((source) => {
		const ok = collect(runs, source, intent).filter(isSuccess);
		const domains = new Set<string>();
		for (const c of ok)
			for (const d of domainsOfCapture(c)) if (d) domains.add(d);
		return {
			source,
			successCount: ok.length,
			meanCitations: mean(ok.map((c) => c.sources.length)),
			domains: [...domains].sort(),
		};
	});
}

// ---------------------------------------------------------------------------
// Share of Voice (text mentions AND citation domains — kept strictly separate)
// ---------------------------------------------------------------------------

interface SoVRow {
	brand: string;
	rate: number; // presence rate over successful captures
	shareOfVoice: number; // share among denominator brands
	meanFirstRank: number | null; // 1-based, among denominator brands present
}

/** SoV rows for one source, from a per-capture "brands present in order" fn. */
function sovTable(
	captures: CapturedResponse[],
	dict: BrandEntry[],
	brandsInCapture: (c: CapturedResponse) => string[],
): SoVRow[] {
	const denom = denominatorBrands(dict);
	const denomSet = new Set(denom);
	const n = captures.length;
	const presence = new Map<string, number>();
	const rankSum = new Map<string, { sum: number; count: number }>();

	for (const c of captures) {
		const ordered = brandsInCapture(c).filter((b) => denomSet.has(b));
		ordered.forEach((b, idx) => {
			presence.set(b, (presence.get(b) ?? 0) + 1);
			const r = rankSum.get(b) ?? { sum: 0, count: 0 };
			r.sum += idx + 1;
			r.count += 1;
			rankSum.set(b, r);
		});
	}

	const totalMentions = denom.reduce((s, b) => s + (presence.get(b) ?? 0), 0);
	return denom.map((brand) => {
		const p = presence.get(brand) ?? 0;
		const r = rankSum.get(brand);
		return {
			brand,
			rate: n ? p / n : 0,
			shareOfVoice: totalMentions ? p / totalMentions : 0,
			meanFirstRank: r?.count ? r.sum / r.count : null,
		};
	});
}

function textSoV(captures: CapturedResponse[], dict: BrandEntry[]): SoVRow[] {
	return sovTable(captures, dict, (c) =>
		resolveBrandMentions(c.response, dict),
	);
}

function citationSoV(
	captures: CapturedResponse[],
	dict: BrandEntry[],
): SoVRow[] {
	return sovTable(captures, dict, (c) =>
		resolveCitationBrands(domainsOfCapture(c), dict),
	);
}

/** Spearman rank correlation of two SoV tables over the same brand set. */
function rankingAgreement(a: SoVRow[], b: SoVRow[]): number | null {
	const brands = a.map((r) => r.brand);
	if (brands.length < 2) return null;
	const rankMap = (rows: SoVRow[]): Map<string, number> => {
		const ordered = [...rows].sort(
			(x, y) =>
				y.shareOfVoice - x.shareOfVoice || x.brand.localeCompare(y.brand),
		);
		const m = new Map<string, number>();
		ordered.forEach((r, i) => m.set(r.brand, i + 1));
		return m;
	};
	const ra = rankMap(a);
	const rb = rankMap(b);
	const n = brands.length;
	let d2 = 0;
	for (const brand of brands) {
		const d = (ra.get(brand) ?? 0) - (rb.get(brand) ?? 0);
		d2 += d * d;
	}
	return 1 - (6 * d2) / (n * (n * n - 1));
}

// ---------------------------------------------------------------------------
// Cost / reliability
// ---------------------------------------------------------------------------

interface CostStat {
	source: ResponseSource;
	total: number;
	failures: number;
	meanDurationMs: number | null;
}

function costStats(runs: RunData[]): CostStat[] {
	return SOURCES.map((source) => {
		const all = collect(runs, source);
		const durations = all
			.map((c) => c.durationMs)
			.filter((d): d is number => typeof d === "number");
		return {
			source,
			total: all.length,
			failures: all.filter((c) => !!c.error).length,
			meanDurationMs: durations.length ? mean(durations) : null,
		};
	});
}

interface WebReliability {
	samples: number;
	failed: number;
	retries: number;
	recycles: number;
	loggedOut: number;
	browsedRate: number;
}

function webReliability(runs: RunData[]): WebReliability | null {
	const metas = runs.flatMap((r) => r.webSamples ?? []);
	if (metas.length === 0) return null;
	const ok = metas.filter((m) => m.ok);
	return {
		samples: metas.length,
		failed: metas.filter((m) => !m.ok).length,
		retries: metas.reduce((s, m) => s + Math.max(0, m.attempts - 1), 0),
		recycles: metas.filter((m) => m.recycledBrowser).length,
		loggedOut: metas.filter((m) => m.errorType === "logged_out").length,
		browsedRate: ok.length ? ok.filter((m) => m.browsed).length / ok.length : 0,
	};
}

/** Sequential wall-time to run 100 prompts × 5 engines at the source's mean pace. */
function throughputHours(meanDurationMs: number | null): number | null {
	if (meanDurationMs == null) return null;
	return (100 * 5 * (meanDurationMs / 1000)) / 3600;
}

// ---------------------------------------------------------------------------
// Cross-run stability (join on prompt text, never array index)
// ---------------------------------------------------------------------------

interface CrossRunStat {
	promptsJoined: number;
	browseStability: number | null; // fraction of prompts where web browse decision agrees
	brandSetJaccard: { source: ResponseSource; jaccard: number | null }[];
	citationJaccard: { source: ResponseSource; jaccard: number | null }[];
}

interface PerPromptSource {
	brands: Set<string>;
	domains: Set<string>;
	browsed: boolean;
}

function indexRun(
	run: RunData,
	dict: BrandEntry[],
): Map<string, Map<ResponseSource, PerPromptSource>> {
	const denom = new Set(denominatorBrands(dict));
	const byPrompt = new Map<string, Map<ResponseSource, PerPromptSource>>();
	for (const result of run.results) {
		const bySource = new Map<ResponseSource, PerPromptSource>();
		for (const source of SOURCES) {
			const ok = result.captures.filter(
				(c) => c.source === source && isSuccess(c),
			);
			if (ok.length === 0) continue;
			const brands = new Set<string>();
			const domains = new Set<string>();
			let browsed = false;
			for (const c of ok) {
				for (const b of resolveBrandMentions(c.response, dict)) {
					if (denom.has(b)) brands.add(b);
				}
				for (const d of domainsOfCapture(c)) if (d) domains.add(d);
				if (c.sources.length > 0) browsed = true;
			}
			bySource.set(source, { brands, domains, browsed });
		}
		byPrompt.set(result.prompt, bySource);
	}
	return byPrompt;
}

function crossRunStability(
	runs: RunData[],
	dict: BrandEntry[],
): CrossRunStat | null {
	if (runs.length < 2) return null;
	const indexed = runs.map((r) => indexRun(r, dict));
	// Prompts present (as web or any source) in at least two runs.
	const promptCounts = new Map<string, number>();
	for (const idx of indexed) {
		for (const prompt of idx.keys()) {
			promptCounts.set(prompt, (promptCounts.get(prompt) ?? 0) + 1);
		}
	}
	const joined = [...promptCounts.entries()]
		.filter(([, n]) => n >= 2)
		.map(([p]) => p);
	if (joined.length === 0) return null;

	// Browse-decision stability (web): all runs that have this prompt agree.
	let stable = 0;
	let browseComparable = 0;
	for (const prompt of joined) {
		const decisions = indexed
			.map((idx) => idx.get(prompt)?.get("web")?.browsed)
			.filter((b): b is boolean => typeof b === "boolean");
		if (decisions.length < 2) continue;
		browseComparable++;
		if (decisions.every((b) => b === decisions[0])) stable++;
	}

	const pairJaccard = (
		source: ResponseSource,
		pick: (p: PerPromptSource) => Set<string>,
	): number | null => {
		const values: number[] = [];
		for (const prompt of joined) {
			const sets = indexed
				.map((idx) => idx.get(prompt)?.get(source))
				.filter((p): p is PerPromptSource => p !== undefined)
				.map(pick);
			for (let i = 0; i < sets.length; i++) {
				for (let j = i + 1; j < sets.length; j++) {
					values.push(jaccard(sets[i] as Set<string>, sets[j] as Set<string>));
				}
			}
		}
		return values.length ? mean(values) : null;
	};

	return {
		promptsJoined: joined.length,
		browseStability: browseComparable ? stable / browseComparable : null,
		brandSetJaccard: SOURCES.map((source) => ({
			source,
			jaccard: pairJaccard(source, (p) => p.brands),
		})),
		citationJaccard: SOURCES.map((source) => ({
			source,
			jaccard: pairJaccard(source, (p) => p.domains),
		})),
	};
}

// ---------------------------------------------------------------------------
// Semantic calibration (optional; only when a run captured embeddings)
// ---------------------------------------------------------------------------

interface SemanticStat {
	floor: { source: ResponseSource; cosine: number | null }[];
	ceiling: { source: ResponseSource; cosine: number | null }[];
	crossWebGrounded: number | null;
}

async function semanticCalibration(
	runs: RunData[],
): Promise<SemanticStat | null> {
	if (!runs.some((r) => r.options.embeddings)) return null;
	const results = runs.flatMap((r) => r.results);
	const respsFor = (result: PromptResult, source: ResponseSource): string[] =>
		result.captures
			.filter((c) => c.source === source && isSuccess(c))
			.map((c) => c.response);

	const allTexts: string[] = [];
	for (const result of results) {
		for (const c of result.captures)
			if (isSuccess(c)) allTexts.push(c.response);
	}
	if (allTexts.length === 0) return null;

	// Ceiling and cross-source are computed PER PROMPT then averaged (matching
	// report.ts), so a prompt's within-source spread isn't diluted by pairs from
	// other prompts. Prompts that can't form a pair (a source with <2 usable
	// samples → meanPairwiseCosine null; an empty side → meanCrossCosine null)
	// contribute nothing and are skipped — never a divide-by-zero or spurious 1.0.
	const avg = (xs: (number | null)[]): number | null => {
		const present = xs.filter((x): x is number => x !== null);
		return present.length ? mean(present) : null;
	};

	try {
		const cache = await embedTexts([...allTexts, UNRELATED_CONTROL]);
		const floor = SOURCES.map((source) => ({
			source,
			cosine: avg(
				results.map((result) => {
					const cosines = respsFor(result, source)
						.map((t) => cachedCosine(cache, t, UNRELATED_CONTROL))
						.filter((v): v is number => v !== null);
					return cosines.length ? mean(cosines) : null;
				}),
			),
		}));
		const ceiling = SOURCES.map((source) => ({
			source,
			cosine: avg(
				results.map((result) =>
					meanPairwiseCosine(cache, respsFor(result, source)),
				),
			),
		}));
		const crossWebGrounded = avg(
			results.map((result) =>
				meanCrossCosine(
					cache,
					respsFor(result, "web"),
					respsFor(result, "api-grounded"),
				),
			),
		);
		return { floor, ceiling, crossWebGrounded };
	} catch (err) {
		console.error(`[analysis] embedding calibration skipped: ${String(err)}`);
		return null;
	}
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

function intentsPresent(runs: RunData[]): Intent[] {
	const set = new Set<Intent>();
	for (const run of runs) {
		for (const r of run.results) set.add(r.intent ?? "unspecified");
	}
	const order: Intent[] = [
		"bilgi_alma",
		"karsilastirma",
		"satin_alma",
		"marka_arama",
		"unspecified",
	];
	return order.filter((i) => set.has(i));
}

function targetBrand(dict: BrandEntry[]): string | null {
	return dict.find((e) => e.role === "target")?.canonical ?? null;
}

function sovMarkdown(title: string, rows: SoVRow[]): string[] {
	const lines = [
		`**${title}**`,
		"",
		"| Brand | mention rate | SoV | mean first rank |",
		"|---|---|---|---|",
	];
	for (const r of rows) {
		lines.push(
			`| ${r.brand} | ${pct(r.rate)} | ${pct(r.shareOfVoice)} | ${
				r.meanFirstRank == null ? "—" : fmt(r.meanFirstRank, 1)
			} |`,
		);
	}
	lines.push("");
	return lines;
}

export async function buildEvaluationAnalysis(
	runs: RunData[],
	opts: EvalOptions,
): Promise<EvaluationReport> {
	const dict = opts.brandDict;
	const denom = denominatorBrands(dict);
	const target = targetBrand(dict);
	const intents = intentsPresent(runs);

	const cost = costStats(runs);
	const web = webReliability(runs);
	const cite = citationStats(runs);
	const semantic = await semanticCalibration(runs);

	// Overall SoV tables per source.
	const overallText = new Map<ResponseSource, SoVRow[]>();
	const overallCite = new Map<ResponseSource, SoVRow[]>();
	for (const source of SOURCES) {
		const caps = collect(runs, source).filter(isSuccess);
		overallText.set(source, textSoV(caps, dict));
		overallCite.set(source, citationSoV(caps, dict));
	}
	const textRankAgreement = rankingAgreement(
		overallText.get("web") ?? [],
		overallText.get("api-grounded") ?? [],
	);

	const cross = crossRunStability(runs, dict);

	// ---- Markdown ----
	const md: string[] = ["# Web-vs-API Worth-It Evaluation — Turkcell", ""];
	md.push(
		`_Generated ${new Date().toISOString()} · ${runs.length} run(s) · ` +
			`${runs.reduce((s, r) => s + r.results.length, 0)} prompt-slot(s)_`,
		"",
	);
	md.push(
		`**Target:** ${target ?? "—"} · **Denominator:** ${denom.join(", ")}`,
		"",
		"> Two SoV signals below — **text mentions** and **cited domains** — are reported",
		"> separately and must **never** be summed.",
		"",
	);

	// 1. Cost / reliability
	md.push("## 1. Cost & reliability delta", "");
	md.push(
		"| Source | samples | failures | mean duration | 100×5 projection |",
		"|---|---|---|---|---|",
	);
	for (const s of cost) {
		const hrs = throughputHours(s.meanDurationMs);
		md.push(
			`| ${s.source} | ${s.total} | ${s.failures} | ${
				s.meanDurationMs == null ? "—" : `${fmt(s.meanDurationMs / 1000, 1)}s`
			} | ${hrs == null ? "—" : `${fmt(hrs, 1)} h/day`} |`,
		);
	}
	md.push("");
	md.push(
		"_Projection = 100 prompts × 5 engines run sequentially at the source's mean pace._",
		"",
	);
	if (web) {
		md.push(
			`**Web leg:** ${web.samples} samples · ${web.failed} failed · ${web.retries} retries · ` +
				`${web.recycles} browser recycles · ${web.loggedOut} logged_out · ` +
				`${pct(web.browsedRate)} browsed.`,
			"",
		);
	} else {
		md.push("_No web-sample diagnostics present in these runs._", "");
	}

	// 2. Citation
	md.push("## 2. Citation delta", "");
	md.push("| Source | mean citations | # domains |", "|---|---|---|");
	for (const c of cite)
		md.push(`| ${c.source} | ${fmt(c.meanCitations)} | ${c.domains.length} |`);
	md.push("");
	const webDomains = new Set(
		cite.find((c) => c.source === "web")?.domains ?? [],
	);
	const grDomains = new Set(
		cite.find((c) => c.source === "api-grounded")?.domains ?? [],
	);
	const shared = [...webDomains].filter((d) => grDomains.has(d)).sort();
	const webOnly = [...webDomains].filter((d) => !grDomains.has(d)).sort();
	const grOnly = [...grDomains].filter((d) => !webDomains.has(d)).sort();
	md.push(
		`- Domain Jaccard (web ↔ api-grounded): **${fmt(jaccard(webDomains, grDomains))}**`,
		`- Shared domains (${shared.length}): ${shared.join(", ") || "—"}`,
		`- Web-only domains (${webOnly.length}): ${webOnly.join(", ") || "—"}`,
		`- Grounded-only domains (${grOnly.length}): ${grOnly.join(", ") || "—"}`,
		"",
	);

	// 3. Share of voice
	md.push("## 3. Share-of-Voice delta", "");
	for (const source of SOURCES) {
		md.push(`### ${source}`, "");
		md.push(...sovMarkdown("Text mentions", overallText.get(source) ?? []));
		md.push(
			...sovMarkdown(
				"Cited domains (separate signal — do not sum)",
				overallCite.get(source) ?? [],
			),
		);
	}
	md.push(
		`- Text-mention ranking agreement (web ↔ api-grounded, Spearman ρ): **${
			textRankAgreement == null ? "—" : fmt(textRankAgreement)
		}**`,
		"",
	);

	// Per-intent condensed (target brand + mean citations)
	if (
		intents.length > 1 ||
		(intents.length === 1 && intents[0] !== "unspecified")
	) {
		md.push("## 4. Per-intent summary", "");
		md.push(
			`Target (${target ?? "—"}) text SoV and mean citations, by intent × source.`,
			"",
			"| Intent | Source | target SoV | mean citations |",
			"|---|---|---|---|",
		);
		for (const intent of intents) {
			for (const source of SOURCES) {
				const caps = collect(runs, source, intent).filter(isSuccess);
				if (caps.length === 0) continue;
				const rows = textSoV(caps, dict);
				const t = target ? rows.find((r) => r.brand === target) : undefined;
				const cites = mean(caps.map((c) => c.sources.length));
				md.push(
					`| ${intent} | ${source} | ${t ? pct(t.shareOfVoice) : "—"} | ${fmt(cites)} |`,
				);
			}
		}
		md.push("");
	}

	// Semantic calibration
	if (semantic) {
		md.push("## 5. Semantic calibration (embeddings)", "");
		md.push(
			"Computed per prompt, then averaged. Floor = mean cosine vs an unrelated " +
				"control; ceiling = within-source mean pairwise (same prompt). Cross-source " +
				"sits between the two.",
			"",
			"| Source | floor (vs control) | ceiling (within) |",
			"|---|---|---|",
		);
		for (const source of SOURCES) {
			const f = semantic.floor.find((x) => x.source === source)?.cosine;
			const c = semantic.ceiling.find((x) => x.source === source)?.cosine;
			md.push(
				`| ${source} | ${f == null ? "—" : fmt(f)} | ${c == null ? "—" : fmt(c)} |`,
			);
		}
		md.push(
			"",
			`- Cross-source cosine (web ↔ api-grounded): **${
				semantic.crossWebGrounded == null ? "—" : fmt(semantic.crossWebGrounded)
			}**`,
			"",
		);
	}

	// Cross-run stability
	if (cross) {
		md.push("## 6. Cross-run stability", "");
		md.push(
			`Joined **${cross.promptsJoined}** prompt(s) across runs by prompt text.`,
			"",
			`- Web browse-decision stability: **${
				cross.browseStability == null ? "—" : pct(cross.browseStability)
			}** of prompts agree across runs.`,
			"",
			"| Source | brand-set Jaccard | citation-domain Jaccard |",
			"|---|---|---|",
		);
		for (const source of SOURCES) {
			const bj = cross.brandSetJaccard.find(
				(x) => x.source === source,
			)?.jaccard;
			const cj = cross.citationJaccard.find(
				(x) => x.source === source,
			)?.jaccard;
			md.push(
				`| ${source} | ${bj == null ? "—" : fmt(bj)} | ${cj == null ? "—" : fmt(cj)} |`,
			);
		}
		md.push("");
	} else if (runs.length < 2) {
		md.push(
			"## 6. Cross-run stability",
			"",
			"_Needs ≥2 runs — supply a second run to `--analyze`._",
			"",
		);
	}

	// ---- JSON ----
	const jsonObj = {
		generatedAt: new Date().toISOString(),
		runs: runs.length,
		target,
		denominator: denom,
		cost: cost.map((c) => ({
			...c,
			projectionHoursPerDay: throughputHours(c.meanDurationMs),
		})),
		webReliability: web,
		citation: cite,
		domainDelta: {
			jaccard: jaccard(webDomains, grDomains),
			shared,
			webOnly,
			grOnly,
		},
		soV: {
			text: Object.fromEntries(overallText),
			citation: Object.fromEntries(overallCite),
			textRankAgreement,
		},
		semantic,
		crossRun: cross,
	};

	// ---- CSV (long-format SoV, both tables flagged) ----
	const csvRows: string[] = [
		"source,table,brand,mention_rate,share_of_voice,mean_first_rank",
	];
	for (const source of SOURCES) {
		const push = (table: string, rows: SoVRow[]) => {
			for (const r of rows) {
				csvRows.push(
					[
						source,
						table,
						r.brand,
						fmt(r.rate, 4),
						fmt(r.shareOfVoice, 4),
						r.meanFirstRank == null ? "" : fmt(r.meanFirstRank, 2),
					].join(","),
				);
			}
		};
		push("text", overallText.get(source) ?? []);
		push("citation", overallCite.get(source) ?? []);
	}

	return {
		markdown: md.join("\n"),
		json: JSON.stringify(jsonObj, null, 2),
		csv: csvRows.join("\n"),
	};
}
