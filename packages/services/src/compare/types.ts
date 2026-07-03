import type { Source } from "@oneglanse/types";

/** The three response sources compared in Phase 1. */
export type ResponseSource = "web" | "api-raw" | "api-grounded";

/** API browsing modes. `raw` = no tools; `grounded` = web_search tool on. */
export type ApiMode = "raw" | "grounded";

/** A single captured response (one sample from one source). */
export interface CapturedResponse {
	source: ResponseSource;
	prompt: string;
	/** 1-based sample index. */
	sample: number;
	response: string;
	sources: Source[];
	/** Set when this sample failed; response/sources are then empty. */
	error?: string;
}

/** All captures for one prompt, across sources and samples. */
export interface PromptResult {
	prompt: string;
	captures: CapturedResponse[];
}

/** Options controlling report generation. */
export interface ReportOptions {
	brands: string[];
	model: string;
	samples: number;
	modes: ApiMode[];
	/** Whether embedding cosine similarity was computed. */
	embeddings: boolean;
}
