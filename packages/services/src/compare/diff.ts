import { createHash } from "node:crypto";
import type { Source } from "@oneglanse/types";
import { env } from "../env.js";
import { azureOpenai } from "../llm/azure.js";

/** Deterministic per-response metrics — no AI judging. */
export interface ResponseMetrics {
	chars: number;
	words: number;
	citationCount: number;
	/** Unique cited domains, sorted. */
	domains: string[];
	/** Brands found in the response, ordered by first mention. */
	brandsMentioned: string[];
	hasNumberedList: boolean;
	hasHeadings: boolean;
}

const NUMBERED_LIST = /^\s*\d+[.)]\s+\S/m;
const MARKDOWN_HEADING = /^\s*#{1,6}\s+\S/m;

/** Lowercased alphanumeric word tokens. */
export function tokenize(text: string): string[] {
	const matches = text.toLowerCase().match(/[a-z0-9]+/g);
	return matches ?? [];
}

/** Jaccard index of two sets of strings (0 when both empty). */
export function jaccard(a: Iterable<string>, b: Iterable<string>): number {
	const setA = new Set(a);
	const setB = new Set(b);
	if (setA.size === 0 && setB.size === 0) return 0;
	let intersection = 0;
	for (const item of setA) {
		if (setB.has(item)) intersection++;
	}
	const union = setA.size + setB.size - intersection;
	return union === 0 ? 0 : intersection / union;
}

/** Lexical similarity: Jaccard over the two token sets. */
export function lexicalSimilarity(a: string, b: string): number {
	return jaccard(tokenize(a), tokenize(b));
}

/** Unique, sorted cited domains from a source list. */
export function domainsOf(sources: Source[]): string[] {
	const set = new Set<string>();
	for (const src of sources) {
		if (src.domain) set.add(src.domain);
	}
	return [...set].sort();
}

/** Brands present in the response, ordered by first appearance (case-insensitive). */
function brandsInOrder(response: string, brands: string[]): string[] {
	const haystack = response.toLowerCase();
	return brands
		.map((brand) => ({ brand, index: haystack.indexOf(brand.toLowerCase()) }))
		.filter((entry) => entry.index >= 0)
		.sort((a, b) => a.index - b.index)
		.map((entry) => entry.brand);
}

export function computeMetrics(
	response: string,
	sources: Source[],
	brands: string[],
): ResponseMetrics {
	return {
		chars: response.length,
		words: tokenize(response).length,
		citationCount: sources.length,
		domains: domainsOf(sources),
		brandsMentioned: brandsInOrder(response, brands),
		hasNumberedList: NUMBERED_LIST.test(response),
		hasHeadings: MARKDOWN_HEADING.test(response),
	};
}

/** Cosine similarity of two equal-length numeric vectors. */
export function cosine(a: number[], b: number[]): number {
	let dot = 0;
	let normA = 0;
	let normB = 0;
	for (let i = 0; i < a.length; i++) {
		const x = a[i] ?? 0;
		const y = b[i] ?? 0;
		dot += x * y;
		normA += x * x;
		normB += y * y;
	}
	if (normA === 0 || normB === 0) return 0;
	return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** SHA-256 of the trimmed text — cache key so identical texts embed once. */
export function hashText(text: string): string {
	return createHash("sha256").update(text.trim()).digest("hex");
}

/** Max inputs per embeddings request (well within the API limit). */
const EMBED_BATCH = 96;

/**
 * Embeds many texts via the configured Azure embedding deployment, batching one
 * array per request and deduplicating identical texts. Returns vectors keyed by
 * content hash — the per-run embedding cache. A math distance, not an AI judge.
 */
export async function embedTexts(texts: string[]): Promise<Map<string, number[]>> {
	const cache = new Map<string, number[]>();
	const unique = [...new Set(texts.map((t) => t.trim()).filter((t) => t.length > 0))];
	for (let i = 0; i < unique.length; i += EMBED_BATCH) {
		const batch = unique.slice(i, i + EMBED_BATCH);
		const res = await azureOpenai().embeddings.create({
			model: env.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
			input: batch,
		});
		res.data.forEach((d, j) => {
			const text = batch[j];
			if (text) cache.set(hashText(text), d.embedding);
		});
	}
	return cache;
}

function vectorsFor(cache: Map<string, number[]>, texts: string[]): number[][] {
	return texts
		.map((t) => cache.get(hashText(t)))
		.filter((v): v is number[] => v !== undefined);
}

/** Cosine between two texts using precomputed vectors; null if either missing. */
export function cachedCosine(
	cache: Map<string, number[]>,
	a: string,
	b: string,
): number | null {
	const [va] = vectorsFor(cache, [a]);
	const [vb] = vectorsFor(cache, [b]);
	return va && vb ? cosine(va, vb) : null;
}

/** Mean cosine over all unordered pairs within one set; null if <2 vectors. */
export function meanPairwiseCosine(
	cache: Map<string, number[]>,
	texts: string[],
): number | null {
	const vecs = vectorsFor(cache, texts);
	if (vecs.length < 2) return null;
	let sum = 0;
	let n = 0;
	for (let i = 0; i < vecs.length; i++) {
		for (let j = i + 1; j < vecs.length; j++) {
			sum += cosine(vecs[i] as number[], vecs[j] as number[]);
			n++;
		}
	}
	return sum / n;
}

/** Mean cosine over all cross pairs between two sets; null if either is empty. */
export function meanCrossCosine(
	cache: Map<string, number[]>,
	textsA: string[],
	textsB: string[],
): number | null {
	const va = vectorsFor(cache, textsA);
	const vb = vectorsFor(cache, textsB);
	if (va.length === 0 || vb.length === 0) return null;
	let sum = 0;
	for (const x of va) for (const y of vb) sum += cosine(x, y);
	return sum / (va.length * vb.length);
}

/**
 * Embedding cosine similarity of two texts via the Azure embedding deployment.
 * Convenience wrapper over embedTexts; returns 0 if either text is empty.
 */
export async function embeddingSimilarity(a: string, b: string): Promise<number> {
	if (!a.trim() || !b.trim()) return 0;
	const cache = await embedTexts([a, b]);
	return cachedCosine(cache, a, b) ?? 0;
}
