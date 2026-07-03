import {
	computeMetrics,
	embedTexts,
	jaccard,
	lexicalSimilarity,
	meanCrossCosine,
	meanPairwiseCosine,
	type ResponseMetrics,
} from "./diff.js";
import type {
	CapturedResponse,
	PromptResult,
	ReportOptions,
	ResponseSource,
} from "./types.js";

/** Column order for source-keyed tables. */
const SOURCE_ORDER: ResponseSource[] = ["web", "api-raw", "api-grounded"];

/** Pairs compared for similarity (both sides must be present to appear). */
const SIMILARITY_PAIRS: [ResponseSource, ResponseSource][] = [
	["web", "api-grounded"],
	["web", "api-raw"],
	["api-grounded", "api-raw"],
];

interface SourceAggregate {
	source: ResponseSource;
	successCount: number;
	errorCount: number;
	meanChars: number;
	meanWords: number;
	meanCitations: number;
	charsVariance: number;
	unionDomains: string[];
	brands: string[];
	numberedListRate: number;
	headingsRate: number;
}

interface PairComparison {
	a: ResponseSource;
	b: ResponseSource;
	domainJaccard: number;
	lexicalJaccard: number;
}

type EmbeddingCache = Map<string, number[]>;

/** Embedding cosine: within-source noise floor + cross-source pair means. */
interface EmbeddingStats {
	/** Mean pairwise cosine among a source's own samples (null if <2). */
	within: { source: ResponseSource; cosine: number | null }[];
	/** Mean cosine across all sample pairs between two sources. */
	cross: { a: ResponseSource; b: ResponseSource; cosine: number | null }[];
}

interface PromptReport {
	prompt: string;
	sources: SourceAggregate[];
	comparisons: PairComparison[];
	embedding?: EmbeddingStats;
}

interface ComparisonReport {
	markdown: string;
	json: string;
	csv: string;
}

function mean(values: number[]): number {
	if (values.length === 0) return 0;
	return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function variance(values: number[]): number {
	if (values.length < 2) return 0;
	const m = mean(values);
	return mean(values.map((v) => (v - m) ** 2));
}

function successful(captures: CapturedResponse[]): CapturedResponse[] {
	return captures.filter((c) => !c.error && c.response.trim().length > 0);
}

function capturesFor(result: PromptResult, source: ResponseSource): CapturedResponse[] {
	return result.captures.filter((c) => c.source === source);
}

function presentSources(results: PromptResult[]): ResponseSource[] {
	const present = new Set<ResponseSource>();
	for (const result of results) {
		for (const capture of result.captures) present.add(capture.source);
	}
	return SOURCE_ORDER.filter((s) => present.has(s));
}

function aggregateSource(
	captures: CapturedResponse[],
	source: ResponseSource,
	brands: string[],
): SourceAggregate {
	const ok = successful(captures);
	const metrics: ResponseMetrics[] = ok.map((c) =>
		computeMetrics(c.response, c.sources, brands),
	);
	const chars = metrics.map((m) => m.chars);

	const unionDomains = new Set<string>();
	const unionBrands = new Set<string>();
	for (const m of metrics) {
		for (const d of m.domains) unionDomains.add(d);
		for (const b of m.brandsMentioned) unionBrands.add(b);
	}

	return {
		source,
		successCount: ok.length,
		errorCount: captures.length - ok.length,
		meanChars: mean(chars),
		meanWords: mean(metrics.map((m) => m.words)),
		meanCitations: mean(metrics.map((m) => m.citationCount)),
		charsVariance: variance(chars),
		unionDomains: [...unionDomains].sort(),
		brands: [...unionBrands],
		numberedListRate: mean(metrics.map((m) => (m.hasNumberedList ? 1 : 0))),
		headingsRate: mean(metrics.map((m) => (m.hasHeadings ? 1 : 0))),
	};
}

/** First successful capture for a source, if any. */
function representative(
	result: PromptResult,
	source: ResponseSource,
): CapturedResponse | undefined {
	return successful(capturesFor(result, source))[0];
}

function comparePair(
	a: CapturedResponse,
	b: CapturedResponse,
): Pick<PairComparison, "domainJaccard" | "lexicalJaccard"> {
	const domainsA = a.sources.map((s) => s.domain).filter((d): d is string => !!d);
	const domainsB = b.sources.map((s) => s.domain).filter((d): d is string => !!d);
	return {
		domainJaccard: jaccard(domainsA, domainsB),
		lexicalJaccard: lexicalSimilarity(a.response, b.response),
	};
}

/** Successful response texts for a source. */
function responsesFor(result: PromptResult, source: ResponseSource): string[] {
	return successful(capturesFor(result, source)).map((c) => c.response);
}

function computeEmbeddingStats(
	result: PromptResult,
	sources: ResponseSource[],
	cache: EmbeddingCache,
): EmbeddingStats {
	const within = sources.map((source) => ({
		source,
		cosine: meanPairwiseCosine(cache, responsesFor(result, source)),
	}));
	const cross: EmbeddingStats["cross"] = [];
	for (const [a, b] of SIMILARITY_PAIRS) {
		if (!sources.includes(a) || !sources.includes(b)) continue;
		cross.push({
			a,
			b,
			cosine: meanCrossCosine(cache, responsesFor(result, a), responsesFor(result, b)),
		});
	}
	return { within, cross };
}

function buildPromptReport(
	result: PromptResult,
	sources: ResponseSource[],
	opts: ReportOptions,
	cache?: EmbeddingCache,
): PromptReport {
	const aggregates = sources.map((source) =>
		aggregateSource(capturesFor(result, source), source, opts.brands),
	);

	const comparisons: PairComparison[] = [];
	for (const [a, b] of SIMILARITY_PAIRS) {
		if (!sources.includes(a) || !sources.includes(b)) continue;
		const repA = representative(result, a);
		const repB = representative(result, b);
		if (!repA || !repB) continue;
		comparisons.push({ a, b, ...comparePair(repA, repB) });
	}

	const embedding =
		opts.embeddings && cache
			? computeEmbeddingStats(result, sources, cache)
			: undefined;

	return { prompt: result.prompt, sources: aggregates, comparisons, embedding };
}

function fmt(n: number, decimals = 0): string {
	return n.toFixed(decimals);
}

function sourceColumn(reports: PromptReport[], source: ResponseSource): SourceAggregate[] {
	return reports
		.map((r) => r.sources.find((s) => s.source === source))
		.filter((s): s is SourceAggregate => s !== undefined);
}

function setsEqual(a: string[], b: string[]): boolean {
	if (a.length !== b.length) return false;
	const setB = new Set(b.map((x) => x.toLowerCase()));
	return a.every((x) => setB.has(x.toLowerCase()));
}

function buildMarkdown(
	reports: PromptReport[],
	sources: ResponseSource[],
	opts: ReportOptions,
): string {
	const lines: string[] = [];
	const header = ["Metric", ...sources];

	lines.push("# Web-vs-API Comparison Report", "");
	lines.push(`_Generated ${new Date().toISOString()}_`, "");
	lines.push(
		`**Model:** ${opts.model} · **Samples/source:** ${opts.samples} · ` +
			`**API modes:** ${opts.modes.join(", ")} · **Embeddings:** ${
				opts.embeddings ? "on" : "off"
			}`,
	);
	if (opts.brands.length > 0) lines.push(`**Brands:** ${opts.brands.join(", ")}`);
	lines.push("");
	lines.push(
		"> Comparison is of **products, not just models**: ChatGPT web = browsing + hidden",
		"> system prompt + memory + product post-processing; the API model is the closest",
		"> configurable match.",
		"",
	);

	// --- Aggregates ---
	lines.push("## Aggregates", "");
	lines.push(`| ${header.join(" | ")} |`);
	lines.push(`|${header.map(() => "---").join("|")}|`);
	const aggRow = (label: string, fn: (col: SourceAggregate[]) => string) =>
		lines.push(`| ${label} | ${sources.map((s) => fn(sourceColumn(reports, s))).join(" | ")} |`);
	aggRow("Mean chars", (col) => fmt(mean(col.map((a) => a.meanChars))));
	aggRow("Mean words", (col) => fmt(mean(col.map((a) => a.meanWords))));
	aggRow("Mean citations", (col) => fmt(mean(col.map((a) => a.meanCitations)), 2));
	aggRow("Length variance (within-source)", (col) =>
		fmt(mean(col.map((a) => a.charsVariance))),
	);
	lines.push("");

	const overlap = (pair: [ResponseSource, ResponseSource], key: keyof PairComparison) =>
		reports
			.map((r) => r.comparisons.find((c) => c.a === pair[0] && c.b === pair[1]))
			.filter((c): c is PairComparison => c !== undefined)
			.map((c) => c[key] as number | null)
			.filter((v): v is number => v !== null);

	const domainOverlaps = overlap(["web", "api-grounded"], "domainJaccard");
	if (domainOverlaps.length > 0) {
		lines.push(
			`- Avg domain overlap (web ↔ api-grounded): ${fmt(mean(domainOverlaps), 2)}`,
		);
	}

	const webCol = sourceColumn(reports, "web");
	const rawCol = sourceColumn(reports, "api-raw");
	if (webCol.length > 0 && rawCol.length > 0) {
		let differ = 0;
		let compared = 0;
		for (const report of reports) {
			const web = report.sources.find((s) => s.source === "web");
			const raw = report.sources.find((s) => s.source === "api-raw");
			if (!web || !raw) continue;
			compared++;
			if (!setsEqual(web.brands, raw.brands)) differ++;
		}
		if (compared > 0) {
			lines.push(
				`- Prompts where brand set differs (web vs api-raw): ${differ}/${compared} ` +
					`(${fmt((differ / compared) * 100)}%)`,
			);
		}
	}

	if (opts.embeddings) {
		const withinAgg = sources
			.map((s) => {
				const vals = reports
					.map((r) => r.embedding?.within.find((w) => w.source === s)?.cosine)
					.filter((v): v is number => v != null);
				return vals.length ? `${s} ${fmt(mean(vals), 2)}` : null;
			})
			.filter((x): x is string => x !== null);
		if (withinAgg.length > 0) {
			lines.push(`- Mean embedding within-source baseline: ${withinAgg.join(" · ")}`);
		}
		const crossAgg = SIMILARITY_PAIRS.map(([a, b]) => {
			const vals = reports
				.map((r) => r.embedding?.cross.find((c) => c.a === a && c.b === b)?.cosine)
				.filter((v): v is number => v != null);
			return vals.length ? `${a} ↔ ${b} ${fmt(mean(vals), 2)}` : null;
		}).filter((x): x is string => x !== null);
		if (crossAgg.length > 0) {
			lines.push(`- Mean embedding cross-source: ${crossAgg.join(" · ")}`);
		}
	}
	lines.push("");

	// --- Per-prompt detail ---
	lines.push("## Per-prompt detail", "");
	reports.forEach((report, i) => {
		lines.push(`### ${i + 1}. "${report.prompt}"`, "");
		lines.push(`| ${header.join(" | ")} |`);
		lines.push(`|${header.map(() => "---").join("|")}|`);
		const cell = (fn: (a: SourceAggregate) => string) =>
			sources
				.map((s) => {
					const agg = report.sources.find((x) => x.source === s);
					return agg ? fn(agg) : "—";
				})
				.join(" | ");
		lines.push(`| Mean chars | ${cell((a) => fmt(a.meanChars))} |`);
		lines.push(`| Mean words | ${cell((a) => fmt(a.meanWords))} |`);
		lines.push(`| Mean citations | ${cell((a) => fmt(a.meanCitations, 2))} |`);
		lines.push(`| # domains | ${cell((a) => fmt(a.unionDomains.length))} |`);
		lines.push(
			`| Domains | ${cell((a) =>
				a.unionDomains.length ? truncateList(a.unionDomains) : "—",
			)} |`,
		);
		lines.push(
			`| Brands (in order) | ${cell((a) => (a.brands.length ? a.brands.join(", ") : "—"))} |`,
		);
		lines.push(`| Numbered list | ${cell((a) => pct(a.numberedListRate))} |`);
		lines.push(`| Headings | ${cell((a) => pct(a.headingsRate))} |`);
		lines.push(`| Failed samples | ${cell((a) => fmt(a.errorCount))} |`);
		lines.push("");

		if (report.comparisons.length > 0) {
			lines.push("**Similarity (representative sample per source)**", "");
			lines.push("| Pair | domain Jaccard | lexical Jaccard |");
			lines.push("|---|---|---|");
			for (const c of report.comparisons) {
				lines.push(
					`| ${c.a} ↔ ${c.b} | ${fmt(c.domainJaccard, 2)} | ${fmt(c.lexicalJaccard, 2)} |`,
				);
			}
			lines.push("");
		}

		if (report.embedding) {
			// within-source baseline = noise floor; cross-source numbers read against it.
			lines.push("**Embedding cosine (mean pairwise across samples)**", "");
			lines.push(`| baseline | ${sources.join(" | ")} |`);
			lines.push(`|---|${sources.map(() => "---").join("|")}|`);
			const cosCell = (v: number | null | undefined) =>
				v == null ? "—" : fmt(v, 2);
			const withinCells = sources.map((s) =>
				cosCell(report.embedding?.within.find((w) => w.source === s)?.cosine),
			);
			lines.push(`| within-source | ${withinCells.join(" | ")} |`, "");
			const crossParts = report.embedding.cross.map(
				(c) => `${c.a} ↔ ${c.b} ${cosCell(c.cosine)}`,
			);
			if (crossParts.length > 0) {
				lines.push(`Cross-source: ${crossParts.join(" · ")}`, "");
			}
		}
	});

	return lines.join("\n");
}

function truncateList(items: string[], max = 6): string {
	if (items.length <= max) return items.join(", ");
	return `${items.slice(0, max).join(", ")} +${items.length - max} more`;
}

function pct(rate: number): string {
	return `${fmt(rate * 100)}%`;
}

function buildCsv(results: PromptResult[], opts: ReportOptions): string {
	const header = [
		"prompt",
		"source",
		"sample",
		"chars",
		"words",
		"citations",
		"domains",
		"brands",
		"numbered_list",
		"headings",
		"error",
	];
	const rows: string[] = [header.join(",")];
	for (const result of results) {
		for (const capture of result.captures) {
			const m = computeMetrics(capture.response, capture.sources, opts.brands);
			rows.push(
				[
					capture.prompt,
					capture.source,
					capture.sample,
					m.chars,
					m.words,
					m.citationCount,
					m.domains.join(" "),
					m.brandsMentioned.join(" "),
					m.hasNumberedList,
					m.hasHeadings,
					capture.error ?? "",
				]
					.map(csvCell)
					.join(","),
			);
		}
	}
	return rows.join("\n");
}

function csvCell(value: string | number | boolean): string {
	const s = String(value);
	return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * Builds the deterministic comparison report from captured responses.
 * Async because embedding similarity (when enabled) calls the embeddings API.
 */
export async function buildComparisonReport(
	results: PromptResult[],
	opts: ReportOptions,
): Promise<ComparisonReport> {
	const sources = presentSources(results);

	// Embed every successful response once (batched + deduped) up front.
	let cache: EmbeddingCache | undefined;
	if (opts.embeddings) {
		const texts: string[] = [];
		for (const result of results) {
			for (const capture of result.captures) {
				if (!capture.error && capture.response.trim()) texts.push(capture.response);
			}
		}
		cache = await embedTexts(texts);
	}

	const reports = results.map((result) =>
		buildPromptReport(result, sources, opts, cache),
	);

	return {
		markdown: buildMarkdown(reports, sources, opts),
		json: JSON.stringify({ options: opts, prompts: reports }, null, 2),
		csv: buildCsv(results, opts),
	};
}
