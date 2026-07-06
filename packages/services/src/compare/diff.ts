import { createHash } from "node:crypto";
import type { Source } from "@oneglanse/types";
import { env } from "../env.js";
import { azureOpenai } from "../llm/azure.js";
import { foldTr } from "./brands.js";

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
export async function embedTexts(
	texts: string[],
): Promise<Map<string, number[]>> {
	const cache = new Map<string, number[]>();
	const unique = [
		...new Set(texts.map((t) => t.trim()).filter((t) => t.length > 0)),
	];
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
export async function embeddingSimilarity(
	a: string,
	b: string,
): Promise<number> {
	if (!a.trim() || !b.trim()) return 0;
	const cache = await embedTexts([a, b]);
	return cachedCosine(cache, a, b) ?? 0;
}

/**
 * Fixed off-topic Turkish sentence used as the semantic **floor** anchor:
 * embedding cosine of any telecom response against this should be low, so a
 * cross-source cosine is read as "distance from unrelated" rather than absolute.
 */
export const UNRELATED_CONTROL =
	"Akdeniz mutfağında zeytinyağlı enginar yemeğinin tarifi ve gerekli malzemeleri nelerdir?";

/** Common Turkish function words (ASCII-folded, matching foldTr output). */
export const TURKISH_STOPWORDS: Set<string> = new Set([
	"ve",
	"ile",
	"veya",
	"ya",
	"yada",
	"hem",
	"ama",
	"fakat",
	"ancak",
	"yani",
	"cunku",
	"ki",
	"de",
	"da",
	"ta",
	"te",
	"mi",
	"mu",
	"bir",
	"bu",
	"su",
	"o",
	"sey",
	"seyler",
	"icin",
	"gibi",
	"kadar",
	"gore",
	"dogru",
	"once",
	"sonra",
	"uzere",
	"daha",
	"cok",
	"az",
	"en",
	"her",
	"hic",
	"tum",
	"butun",
	"bazi",
	"ne",
	"neden",
	"nasil",
	"hangi",
	"kim",
	"nedir",
	"nelerdir",
	"midir",
	"misin",
	"olan",
	"olarak",
	"olur",
	"ise",
	"iken",
	"ben",
	"sen",
	"biz",
	"siz",
	"onlar",
	"bana",
	"sana",
	"ona",
	"bize",
	"size",
	"onlara",
	"benim",
	"senin",
	"onun",
	"bizim",
	"sizin",
]);

// Light Turkish suffix backstop for keyphrases — deliberately shallow to avoid
// over-stripping (which would collide distinct stems). Approximate; longest first.
const TR_SUFFIXES = [
	"larindan",
	"lerinden",
	"lariyla",
	"leriyle",
	"larini",
	"lerini",
	"larin",
	"lerin",
	"lari",
	"leri",
	"lar",
	"ler",
	"dan",
	"den",
	"tan",
	"ten",
	"nin",
	"nun",
	"da",
	"de",
	"ta",
	"te",
	"ya",
	"ye",
	"yi",
	"yu",
	"si",
	"su",
	"in",
	"un",
	"im",
	"um",
	"i",
	"u",
];

/** Strip at most one shallow Turkish suffix, keeping stems of length ≥ 4. */
function lightStem(word: string): string {
	for (const suf of TR_SUFFIXES) {
		if (word.length - suf.length >= 4 && word.endsWith(suf)) {
			return word.slice(0, word.length - suf.length);
		}
	}
	return word;
}

/** Folded content tokens (stopwords and length-1 tokens dropped), in order. */
function contentTokens(text: string): string[] {
	const tokens = foldTr(text).match(/[a-z0-9]+/g) ?? [];
	return tokens.filter(
		(t) => t.length >= 2 && !/^\d+$/.test(t) && !TURKISH_STOPWORDS.has(t),
	);
}

/**
 * Top content keyphrases: stemmed unigrams + adjacent-content bigrams, ranked by
 * frequency (ties broken alphabetically for determinism). Turkish stopwords are
 * filtered and a light suffix stem is applied — the stemming is approximate.
 */
export function extractKeyphrases(text: string, topN = 15): string[] {
	const tokens = contentTokens(text).map(lightStem);
	const counts = new Map<string, number>();
	const bump = (phrase: string) =>
		counts.set(phrase, (counts.get(phrase) ?? 0) + 1);
	for (let i = 0; i < tokens.length; i++) {
		const w = tokens[i] as string;
		bump(w);
		if (i + 1 < tokens.length) bump(`${w} ${tokens[i + 1]}`);
	}
	return [...counts.entries()]
		.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
		.slice(0, topN)
		.map(([phrase]) => phrase);
}

export interface KeyphraseOverlap {
	jaccard: number;
	shared: string[];
	uniqueA: string[];
	uniqueB: string[];
}

/** Jaccard + shared/unique keyphrase lists between two texts' top-N phrases. */
export function keyphraseOverlap(
	a: string,
	b: string,
	topN = 15,
): KeyphraseOverlap {
	const setA = new Set(extractKeyphrases(a, topN));
	const setB = new Set(extractKeyphrases(b, topN));
	const shared = [...setA].filter((p) => setB.has(p)).sort();
	const uniqueA = [...setA].filter((p) => !setB.has(p)).sort();
	const uniqueB = [...setB].filter((p) => !setA.has(p)).sort();
	return { jaccard: jaccard(setA, setB), shared, uniqueA, uniqueB };
}
