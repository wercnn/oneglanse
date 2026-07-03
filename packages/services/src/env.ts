import { z } from "zod";

const asNumber = (fallback: number) =>
	z.preprocess((value) => {
		if (typeof value === "number") return value;
		if (typeof value !== "string") return fallback;
		const trimmed = value.trim();
		if (!trimmed) return fallback;
		const parsed = Number(trimmed);
		return Number.isFinite(parsed) ? parsed : fallback;
	}, z.number());

const ServicesEnvSchema = z.object({
	REDIS_HOST: z.string().trim().default("redis"),
	REDIS_PORT: asNumber(6379).default(6379),
	REDIS_PASSWORD: z.string().optional(),
	API_BASE_URL: z.string().url().optional(),
	INTERNAL_CRON_SECRET: z.string().optional(),
	OPENAI_API_KEY: z.string().optional(),
	ANTHROPIC_API_KEY: z.string().optional(),
	ANALYSIS_LLM_PROVIDER: z.enum(["openai", "claude"]).default("openai"),
	// Azure OpenAI — used by the web-vs-API comparison harness.
	AZURE_OPENAI_ENDPOINT: z.string().trim().optional(),
	AZURE_OPENAI_API_KEY: z.string().trim().optional(),
	AZURE_OPENAI_DEPLOYMENT: z.string().trim().optional(),
	AZURE_OPENAI_API_VERSION: z.string().trim().default("preview"),
	// Embedding model deployment (same resource as the chat model).
	AZURE_OPENAI_EMBEDDING_DEPLOYMENT: z
		.string()
		.trim()
		.default("text-embedding-3-small"),
});

export const env = ServicesEnvSchema.parse(process.env);
