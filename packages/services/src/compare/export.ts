import { computeMetrics } from "./diff.js";
import type { PromptResult, ResponseSource } from "./types.js";

const SOURCE_ORDER: ResponseSource[] = ["web", "api-raw", "api-grounded"];

function csvCell(value: string | number | boolean): string {
	const s = String(value);
	return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * Full-text transcript (markdown): every captured response for every prompt,
 * grouped source → sample, with citations and basic per-response metrics. This
 * is the response archive the RFP requires — nothing is summarised away.
 */
export function buildTranscript(results: PromptResult[]): string {
	const lines: string[] = ["# Response Transcript", ""];
	lines.push(
		`_Generated ${new Date().toISOString()} · ${results.length} prompt(s)_`,
		"",
	);

	results.forEach((result, i) => {
		const tags = [
			`intent: ${result.intent ?? "unspecified"}`,
			result.brand ? `brand: ${result.brand}` : null,
		]
			.filter(Boolean)
			.join(" · ");
		lines.push(`## ${i + 1}. ${result.prompt}`, "", `_${tags}_`, "");

		for (const source of SOURCE_ORDER) {
			const captures = result.captures
				.filter((c) => c.source === source)
				.sort((a, b) => a.sample - b.sample);
			if (captures.length === 0) continue;
			lines.push(`### ${source}`, "");
			for (const c of captures) {
				const m = computeMetrics(c.response, c.sources, []);
				const timing =
					c.durationMs != null ? ` · ${(c.durationMs / 1000).toFixed(1)}s` : "";
				lines.push(`#### sample ${c.sample}${timing}`, "");
				if (c.error) {
					lines.push(
						`> **FAILED** (${c.errorType ?? "error"}): ${c.error}`,
						"",
					);
					continue;
				}
				lines.push(
					`_${m.chars} chars · ${m.words} words · ${m.citationCount} citation(s)_`,
					"",
				);
				if (m.domains.length > 0) {
					lines.push(`**Domains:** ${m.domains.join(", ")}`, "");
				}
				lines.push(c.response.trim() || "_(empty)_", "");
			}
		}
	});

	return lines.join("\n");
}

/**
 * One CSV row per prompt × source × sample, carrying the full response text plus
 * the intent/brand tags and per-sample timing/error. The archive of record for
 * spot-checking and downstream analysis.
 */
export function buildResponsesCsv(results: PromptResult[]): string {
	const header = [
		"prompt",
		"intent",
		"brand",
		"source",
		"sample",
		"chars",
		"words",
		"citations",
		"domains",
		"duration_ms",
		"error",
		"response",
	];
	const rows: string[] = [header.join(",")];
	for (const result of results) {
		const sorted = [...result.captures].sort(
			(a, b) =>
				SOURCE_ORDER.indexOf(a.source) - SOURCE_ORDER.indexOf(b.source) ||
				a.sample - b.sample,
		);
		for (const c of sorted) {
			const m = computeMetrics(c.response, c.sources, []);
			rows.push(
				[
					result.prompt,
					result.intent ?? "unspecified",
					result.brand ?? "",
					c.source,
					c.sample,
					m.chars,
					m.words,
					m.citationCount,
					m.domains.join(" "),
					c.durationMs ?? "",
					c.error ?? "",
					c.response,
				]
					.map(csvCell)
					.join(","),
			);
		}
	}
	return rows.join("\n");
}
