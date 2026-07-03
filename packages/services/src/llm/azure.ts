import { EnvError } from "@oneglanse/errors";
import OpenAI from "openai";
import { env } from "../env.js";

let client: OpenAI | null = null;

/**
 * Normalize an Azure endpoint to the `/openai/v1/` root the SDK expects.
 * Accepts a bare resource endpoint or one that already includes
 * `/openai/v1` and/or a trailing `/responses` path.
 */
function toV1BaseUrl(endpoint: string): string {
	let url = endpoint.trim().replace(/\/+$/, "");
	url = url.replace(/\/responses$/i, "");
	if (!/\/openai\/v1$/i.test(url)) {
		url = `${url.replace(/\/openai(\/v1)?$/i, "")}/openai/v1`;
	}
	return `${url}/`;
}

/**
 * Lazily-constructed Azure OpenAI client for the comparison harness.
 * Uses the Azure v1 API surface: the deployment name is passed as `model`
 * on each call (see captureOpenAiResponse / embeddingSimilarity).
 */
export function azureOpenai(): OpenAI {
	if (client) return client;

	const endpoint = env.AZURE_OPENAI_ENDPOINT;
	const apiKey = env.AZURE_OPENAI_API_KEY;
	if (!endpoint || !apiKey) {
		throw new EnvError(
			"AZURE_OPENAI_ENDPOINT",
			"Missing Azure OpenAI config. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your environment.",
		);
	}

	client = new OpenAI({
		apiKey,
		baseURL: toV1BaseUrl(endpoint),
		defaultQuery: { "api-version": env.AZURE_OPENAI_API_VERSION },
	});
	return client;
}
