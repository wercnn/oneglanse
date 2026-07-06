import type { BrandEntry } from "./types.js";

/**
 * Turkish-aware ASCII fold: lowercases and strips diacritics so brand aliases
 * match regardless of casing/accents.
 *
 * The Turkish dotted/dotless I is the trap `String.prototype.toLowerCase`
 * mishandles: `İ` (U+0130) decomposes to `I` + combining dot, and a naive
 * `I → i` is wrong for Turkish. We normalise (NFD) and strip combining marks —
 * which folds the dot-above of `İ`, plus the cedilla/breve/diaeresis of
 * ç/ş/ğ/ö/ü — then collapse both capital `I` and dotless `ı` to ASCII `i`
 * before a generic lowercase. All four of İ/I/ı/i therefore fold to `i`
 * deterministically.
 */
export function foldTr(s: string): string {
	return s
		.normalize("NFD")
		.replace(/\p{Mn}/gu, "") // combining marks: dot-above, cedilla, breve, diaeresis
		.replace(/[Iı]/g, "i") // Turkish dotless I / ı (and plain capital I) → ASCII i
		.toLowerCase();
}

function escapeRegExp(s: string): string {
	return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Earliest boundary-anchored occurrence of a folded alias in folded text, or
 * -1. Anchored on the left (preceding char must be a non-letter) and
 * suffix-tolerant on the right (trailing letters/apostrophe allowed, so
 * `turkcell` matches `Turkcell'in`). Whitespace inside multiword aliases
 * matches any run of whitespace.
 */
function firstMentionIndex(foldedText: string, foldedAlias: string): number {
	if (!foldedAlias) return -1;
	const pattern = foldedAlias.split(/\s+/).map(escapeRegExp).join("\\s+");
	const re = new RegExp(`(?<!\\p{L})${pattern}`, "u");
	const m = re.exec(foldedText);
	return m ? m.index : -1;
}

/**
 * Canonical brands mentioned in the text, in first-mention order. Each brand's
 * aliases (and its canonical spelling) all collapse to the single canonical —
 * so every Türk Telekom surface form (türk telekom / TT Mobil / TT / Avea)
 * counts once, as one entity.
 */
export function resolveBrandMentions(
	text: string,
	dict: BrandEntry[],
): string[] {
	const folded = foldTr(text);
	const found: { canonical: string; index: number }[] = [];
	for (const entry of dict) {
		let best = -1;
		for (const surface of [entry.canonical, ...entry.aliases]) {
			const idx = firstMentionIndex(folded, foldTr(surface));
			if (idx >= 0 && (best < 0 || idx < best)) best = idx;
		}
		if (best >= 0) found.push({ canonical: entry.canonical, index: best });
	}
	return found.sort((a, b) => a.index - b.index).map((f) => f.canonical);
}

/**
 * Canonical brands cited as a domain, in first-cited order. A cited host counts
 * for a brand when it equals or is a subdomain of one of the brand's `domains`
 * (so `www.turkcell.com.tr` and `turkcell.com.tr` both resolve to Turkcell).
 */
export function resolveCitationBrands(
	domains: (string | null)[],
	dict: BrandEntry[],
): string[] {
	const result: string[] = [];
	const seen = new Set<string>();
	for (const dom of domains) {
		if (!dom) continue;
		const host = dom.toLowerCase();
		for (const entry of dict) {
			if (!entry.domains?.length) continue;
			const hit = entry.domains.some((d) => {
				const base = d.toLowerCase();
				return host === base || host.endsWith(`.${base}`);
			});
			if (hit && !seen.has(entry.canonical)) {
				seen.add(entry.canonical);
				result.push(entry.canonical);
			}
		}
	}
	return result;
}

/** Canonical names that form the SoV denominator: target + competitors only. */
export function denominatorBrands(dict: BrandEntry[]): string[] {
	return dict
		.filter((e) => e.role === "target" || e.role === "competitor")
		.map((e) => e.canonical);
}
