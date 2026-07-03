import {
	computeMetrics,
	embeddingSimilarity,
	jaccard,
	lexicalSimilarity,
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
	embeddingCosine: number | null;
}

interface PromptReport {
	prompt: string;
	sources: SourceAggregate[];
	comparisons: PairComparison[];
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

async function comparePair(
	a: CapturedResponse,
	b: CapturedResponse,
	withEmbeddings: boolean,
): Promise<Pick<PairComparison, "domainJaccard" | "lexicalJaccard" | "embeddingCosine">> {
	const domainsA = a.sources.map((s) => s.domain).filter((d): d is string => !!d);
	const domainsB = b.sources.map((s) => s.domain).filter((d): d is string => !!d);
	return {
		domainJaccard: jaccard(domainsA, domainsB),
		lexicalJaccard: lexicalSimilarity(a.response, b.response),
		embeddingCosine: withEmbeddings
			? await embeddingSimilarity(a.response, b.response)
			: null,
	};
}

async function buildPromptReport(
	result: PromptResult,
	sources: ResponseSource[],
	opts: ReportOptions,
): Promise<PromptReport> {
	const aggregates = sources.map((source) =>
		aggregateSource(capturesFor(result, source), source, opts.brands),
	);

	const comparisons: PairComparison[] = [];
	for (const [a, b] of SIMILARITY_PAIRS) {
		if (!sources.includes(a) || !sources.includes(b)) continue;
		const repA = representative(result, a);
		const repB = representative(result, b);
		if (!repA || !repB) continue;
		const metrics = await comparePair(repA, repB, opts.embeddings);
		comparisons.push({ a, b, ...metrics });
	}

	return { prompt: result.prompt, sources: aggregates, comparisons };
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
		const embWebRaw = overlap(["web", "api-raw"], "embeddingCosine");
		if (embWebRaw.length > 0) {
			lines.push(
				`- Avg embedding similarity (web ↔ api-raw): ${fmt(mean(embWebRaw), 2)}`,
			);
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
			const simHeader = ["Pair", "domain Jaccard", "lexical Jaccard"];
			if (opts.embeddings) simHeader.push("embedding cosine");
			lines.push(`| ${simHeader.join(" | ")} |`);
			lines.push(`|${simHeader.map(() => "---").join("|")}|`);
			for (const c of report.comparisons) {
				const row = [
					`${c.a} ↔ ${c.b}`,
					fmt(c.domainJaccard, 2),
					fmt(c.lexicalJaccard, 2),
				];
				if (opts.embeddings) {
					row.push(c.embeddingCosine === null ? "—" : fmt(c.embeddingCosine, 2));
				}
				lines.push(`| ${row.join(" | ")} |`);
			}
			lines.push("");
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
	const reports: PromptReport[] = [];
	for (const result of results) {
		reports.push(await buildPromptReport(result, sources, opts));
	}

	return {
		markdown: buildMarkdown(reports, sources, opts),
		json: JSON.stringify({ options: opts, prompts: reports }, null, 2),
		csv: buildCsv(results, opts),
	};
}
