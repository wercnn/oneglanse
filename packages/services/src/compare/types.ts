import type { Source } from "@oneglanse/types";

/** The three response sources compared in Phase 1. */
export type ResponseSource = "web" | "api-raw" | "api-grounded";

/** API browsing modes. `raw` = no tools; `grounded` = web_search tool on. */
export type ApiMode = "raw" | "grounded";

/** RFP intent taxonomy for the Turkcell study. `unspecified` = untagged prompt. */
export type Intent =
	| "bilgi_alma"
	| "karsilastirma"
	| "satin_alma"
	| "marka_arama"
	| "unspecified";

/** A brand's role in the SoV denominator. */
export type BrandRole = "target" | "competitor" | "family" | "control";

/** One brand in the SoV dictionary: aliases fold to a single canonical. */
export interface BrandEntry {
	canonical: string;
	aliases: string[];
	role: BrandRole;
	/** Domains that count as this brand when cited (already bare hosts). */
	domains?: string[];
}

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
	/** Wall-clock round-trip for this sample, populated for all three sources. */
	durationMs?: number;
	/** Attempts made before success/failure (web retries; 1 for API). */
	attempts?: number;
	/** classifyError bucket for a failed sample. */
	errorType?: string;
}

/** All captures for one prompt, across sources and samples. */
export interface PromptResult {
	prompt: string;
	captures: CapturedResponse[];
	/** RFP intent tag for this prompt, when the corpus provides one. */
	intent?: Intent;
	/** Target brand this prompt is about (canonical), when tagged. */
	brand?: string;
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
