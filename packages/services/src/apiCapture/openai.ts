import type { Source } from "@oneglanse/types";
import { getDomain } from "@oneglanse/utils";
import type OpenAI from "openai";
import { azureOpenai } from "../llm/azure.js";

export interface CaptureOpenAiOptions {
	prompt: string;
	model: string;
	/** When true, enable the built-in web_search tool and extract citations. */
	grounded: boolean;
}

/**
 * Captures a single OpenAI API response for a prompt.
 *
 *   raw      → plain completion, no tools, answers from training memory (~0 citations).
 *   grounded → same model + the built-in `web_search` tool; the model forms its
 *              own queries and answers with `url_citation` annotations.
 *
 * Returns the same `{ response, sources }` shape as the Camoufox web harness so
 * both feed identical diff/report code.
 */
export async function captureOpenAiResponse({
	prompt,
	model,
	grounded,
}: CaptureOpenAiOptions): Promise<{ response: string; sources: Source[] }> {
	const res = await azureOpenai().responses.create({
		model,
		input: prompt,
		...(grounded ? { tools: [{ type: "web_search" }] } : {}),
	});

	const response = res.output_text?.trim() ?? "";
	const sources = grounded ? extractCitations(res) : [];
	return { response, sources };
}

/** Pulls `url_citation` annotations out of the Responses output, deduped by URL. */
function extractCitations(res: OpenAI.Responses.Response): Source[] {
	const byUrl = new Map<string, Source>();

	for (const item of res.output) {
		if (item.type !== "message") continue;
		for (const part of item.content) {
			if (part.type !== "output_text") continue;
			for (const annotation of part.annotations) {
				if (annotation.type !== "url_citation") continue;
				if (byUrl.has(annotation.url)) continue;
				byUrl.set(annotation.url, {
					title: annotation.title ?? "",
					cited_text: "",
					url: annotation.url,
					domain: getDomain(annotation.url) || null,
					favicon: null,
				});
			}
		}
	}

	return [...byUrl.values()];
}
