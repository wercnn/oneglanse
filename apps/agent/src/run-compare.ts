/**
 * Web-vs-API comparison harness (Phase 1: ChatGPT).
 *
 * Captures, for each prompt in compare.config.json:
 *   - web         : ChatGPT product via Camoufox (fresh chat per sample)
 *   - api-raw     : OpenAI API, no tools (~0 citations)
 *   - api-grounded: OpenAI API + web_search tool (with citations)
 *
 * then writes a deterministic diff report (markdown/json/csv) under
 * apps/agent/compare-output/. No LLM judge — see web-vs-api-comparison-plan.md.
 *
 * Usage: node --loader ts-node/esm src/run-compare.ts [configPath]
 */
import "./env.js";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { classifyError } from "@oneglanse/errors";
import {
	type BrandEntry,
	type CapturedResponse,
	type Intent,
	type PromptResult,
	type ReportOptions,
	type RunData,
	buildComparisonReport,
	buildEvaluationAnalysis,
	buildResponsesCsv,
	buildTranscript,
	captureOpenAiResponse,
	denominatorBrands,
} from "@oneglanse/services";
import { z } from "zod";
import { createAgent } from "./core/createAgent.js";
import { executePrompt } from "./core/prompt-runner/executePrompt.js";
import { resetChatgptPage } from "./core/providers/chatgpt/lib/pageLifecycle.js";

const here = path.dirname(fileURLToPath(import.meta.url));

const IntentEnum = z.enum([
	"bilgi_alma",
	"karsilastirma",
	"satin_alma",
	"marka_arama",
	"unspecified",
]);

const PromptObjectSchema = z.object({
	text: z.string().trim().min(1),
	intent: IntentEnum.default("unspecified"),
	brand: z.string().trim().min(1).optional(),
	targetBrands: z.array(z.string().trim().min(1)).optional(),
});

const BrandEntrySchema = z.object({
	canonical: z.string().trim().min(1),
	aliases: z.array(z.string().trim().min(1)).default([]),
	role: z.enum(["target", "competitor", "family", "control"]),
	domains: z.array(z.string().trim().min(1)).optional(),
});

const ConfigSchema = z.object({
	prompts: z
		.array(z.union([z.string().trim().min(1), PromptObjectSchema]))
		.min(1),
	brands: z.array(z.string().trim().min(1)).default([]),
	brandDict: z.array(BrandEntrySchema).optional(),
	samples: z.number().int().positive().default(3),
	// Deployment name (Azure). Falls back to AZURE_OPENAI_DEPLOYMENT when omitted.
	model: z.string().trim().min(1).optional(),
	modes: z
		.array(z.enum(["raw", "grounded"]))
		.min(1)
		.default(["raw", "grounded"]),
	embeddings: z.boolean().default(false),
});

type Config = z.infer<typeof ConfigSchema>;

/** A prompt normalized to object form, with intent/brand tags. */
interface RunPrompt {
	text: string;
	intent: Intent;
	brand?: string;
	targetBrands?: string[];
}

function normalizePrompts(config: Config): RunPrompt[] {
	return config.prompts.map((p) =>
		typeof p === "string"
			? { text: p, intent: "unspecified" as Intent }
			: {
					text: p.text,
					intent: p.intent,
					brand: p.brand,
					targetBrands: p.targetBrands,
				},
	);
}

/** The SoV brand dictionary: explicit `brandDict`, else legacy `brands` as competitors. */
function resolveBrandDict(config: Config): BrandEntry[] {
	if (config.brandDict?.length) return config.brandDict;
	return config.brands.map((b) => ({
		canonical: b,
		aliases: [b],
		role: "competitor" as const,
	}));
}

function loadConfig(): Config {
	const configPath = process.argv[2]
		? path.resolve(process.cwd(), process.argv[2])
		: path.resolve(here, "..", "compare.config.json");
	if (!fs.existsSync(configPath)) {
		throw new Error(`Config not found: ${configPath}`);
	}
	const raw = JSON.parse(fs.readFileSync(configPath, "utf8"));
	return ConfigSchema.parse(raw);
}

function errMessage(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}

const MAX_ATTEMPTS = 3;
const BACKOFF_MS = [5_000, 15_000, 45_000]; // between attempts; exponential, not jitter
const RECYCLE_EVERY = 5; // proactive browser recycle every N web captures
const RECYCLE_BACKOFF_MS = 15_000; // wait before retrying a failed recycle

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const randomBetween = (min: number, max: number) =>
	Math.floor(min + Math.random() * (max - min));

// Phases we surface on failure, matched against the withTimeout labels / error text.
const KNOWN_PHASES = [
	"beforePromptHook",
	"waitForEditorReady",
	"insertPromptIntoEditor",
	"afterTypingHook",
	"beforeSubmitHook",
	"Submission phase",
	"post-submit stabilization",
	"afterSubmitHook",
	"fetchPromptResponses",
	"extractSources",
	"askPrompt",
	"navigateToPrompt",
];
function extractPhase(msg: string): string | undefined {
	return KNOWN_PHASES.find((p) => msg.includes(p));
}

/** Per-web-sample diagnostics, merged into the report JSON. */
interface WebSampleMeta {
	prompt: string;
	promptIndex: number;
	sample: number;
	ok: boolean;
	attempts: number;
	recycledBrowser: boolean;
	durationMs: number;
	browsed: boolean;
	error?: string;
	errorType?: string;
	phase?: string;
}

type WebPage = Parameters<typeof executePrompt>[0];

async function screenshotFailure(
	page: WebPage,
	failuresDir: string,
	promptIndex: number,
	sample: number,
): Promise<void> {
	try {
		const ts = new Date().toISOString().replace(/[:.]/g, "-");
		const buf = await page.screenshot({ fullPage: true });
		fs.writeFileSync(
			path.join(failuresDir, `${promptIndex}-${sample}-${ts}.png`),
			buf,
		);
	} catch {
		// best effort — a dead page must not crash the run
	}
}

/**
 * ChatGPT web capture with reliability handling:
 *  - interleaved round-robin sampling (round 1 of all prompts, then round 2, …)
 *    so identical prompts are spaced out instead of fired back-to-back,
 *  - per-sample retry (max 3) with 5s/15s/45s backoff + resetChatgptPage between,
 *  - browser recycle: proactively every 5 captures, reactively before a 3rd attempt
 *    (navigation reset alone does not recover a degraded session),
 *  - 2–6s jitter between successful captures to ease rate-limit pressure.
 */
export async function captureWeb(
	prompts: RunPrompt[],
	samples: number,
	failuresDir: string,
	push: (capture: CapturedResponse) => void,
	meta: WebSampleMeta[],
): Promise<void> {
	let agent = await createAgent("chatgpt");
	let sinceRecycle = 0;
	// Set when the ChatGPT session is gone: stop the web leg but fall through to
	// the normal end-of-run write so partial web + full API still persist.
	let stopWeb = false;

	// Recycle the browser (cleanup + fresh agent), retrying once after a backoff.
	// Returns false if a working browser couldn't be launched — a dead browser
	// must never crash the batch, so callers skip the sample and try again later.
	const recycle = async (): Promise<boolean> => {
		await agent.cleanup().catch(() => {});
		for (let r = 1; r <= 2; r++) {
			try {
				agent = await createAgent("chatgpt");
				sinceRecycle = 0;
				return true;
			} catch (err) {
				console.error(
					`[web] recycle attempt ${r}/2 failed: ${errMessage(err)}`,
				);
				if (r < 2) await sleep(RECYCLE_BACKOFF_MS);
			}
		}
		return false;
	};

	// Records the current sample as failed because the browser couldn't recycle.
	const recordRecycleFailure = (
		prompt: string,
		p: number,
		sample: number,
		attempts: number,
		started: number,
	) => {
		const error = "browser recycle failed";
		push({
			source: "web",
			prompt,
			sample,
			response: "",
			sources: [],
			error,
			durationMs: Date.now() - started,
			attempts,
			errorType: "recycle_failed",
		});
		meta.push({
			prompt,
			promptIndex: p,
			sample,
			ok: false,
			attempts,
			recycledBrowser: true,
			durationMs: Date.now() - started,
			browsed: false,
			error,
			errorType: "recycle_failed",
		});
		sinceRecycle = RECYCLE_EVERY; // force a fresh recycle before the next sample
	};

	try {
		for (let sample = 1; sample <= samples; sample++) {
			for (let p = 0; p < prompts.length; p++) {
				const prompt = prompts[p]?.text;
				if (!prompt) continue;
				const started = Date.now();
				let recycledThisSample = false;

				if (sinceRecycle >= RECYCLE_EVERY) {
					console.log("[web] proactive browser recycle");
					if (await recycle()) {
						recycledThisSample = true;
					} else {
						console.error(
							"[web] recycle failed twice — skipping sample, will retry recycle",
						);
						recordRecycleFailure(prompt, p, sample, 0, started);
						continue;
					}
				}
				sinceRecycle++;

				for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
					console.log(
						`[web] "${prompt}" sample ${sample}/${samples} (attempt ${attempt}/${MAX_ATTEMPTS})`,
					);
					try {
						await resetChatgptPage(agent.page);
						const { response, sources } = await executePrompt(
							agent.page,
							prompt,
							"chatgpt",
						);
						push({
							source: "web",
							prompt,
							sample,
							response,
							sources,
							durationMs: Date.now() - started,
							attempts: attempt,
						});
						meta.push({
							prompt,
							promptIndex: p,
							sample,
							ok: true,
							attempts: attempt,
							recycledBrowser: recycledThisSample,
							durationMs: Date.now() - started,
							browsed: sources.length > 0,
						});
						break;
					} catch (err) {
						const msg = errMessage(err);
						const errorType = classifyError(err);
						console.error(`[web] attempt ${attempt} failed: ${msg}`);

						// Session gone: retrying won't recover it. Record this sample,
						// then stop the web leg — but fall through (break, don't return)
						// so the run still reaches its normal end-of-run write.
						if (errorType === "logged_out") {
							console.error(
								"[web] logged_out — stopping web leg, API leg will still run",
							);
							await screenshotFailure(agent.page, failuresDir, p, sample);
							push({
								source: "web",
								prompt,
								sample,
								response: "",
								sources: [],
								error: msg,
								durationMs: Date.now() - started,
								attempts: attempt,
								errorType,
							});
							meta.push({
								prompt,
								promptIndex: p,
								sample,
								ok: false,
								attempts: attempt,
								recycledBrowser: recycledThisSample,
								durationMs: Date.now() - started,
								browsed: false,
								error: msg,
								errorType,
								phase: extractPhase(msg),
							});
							stopWeb = true;
							break;
						}

						if (attempt === MAX_ATTEMPTS) {
							await screenshotFailure(agent.page, failuresDir, p, sample);
							push({
								source: "web",
								prompt,
								sample,
								response: "",
								sources: [],
								error: msg,
								durationMs: Date.now() - started,
								attempts: attempt,
								errorType,
							});
							meta.push({
								prompt,
								promptIndex: p,
								sample,
								ok: false,
								attempts: attempt,
								recycledBrowser: recycledThisSample,
								durationMs: Date.now() - started,
								browsed: false,
								error: msg,
								errorType,
								phase: extractPhase(msg),
							});
							break;
						}

						// After a 2nd failure, a fresh page isn't enough — recycle the browser.
						if (attempt === 2) {
							console.log(
								"[web] reactive browser recycle before final attempt",
							);
							if (await recycle()) {
								recycledThisSample = true;
							} else {
								console.error(
									"[web] recycle failed twice — marking sample failed, continuing run",
								);
								recordRecycleFailure(prompt, p, sample, attempt, started);
								break;
							}
						}
						await sleep(BACKOFF_MS[attempt - 1] ?? 45_000);
					}
				}

				if (stopWeb) break;
				await sleep(randomBetween(2_000, 6_000));
			}
			if (stopWeb) break;
		}
	} finally {
		await agent.cleanup().catch(() => {});
	}
}

/** OpenAI API capture for the configured modes. */
async function captureApi(
	prompts: RunPrompt[],
	samples: number,
	modes: Config["modes"],
	model: string,
	push: (capture: CapturedResponse) => void,
): Promise<void> {
	for (const { text: prompt } of prompts) {
		for (const mode of modes) {
			const source = mode === "grounded" ? "api-grounded" : "api-raw";
			for (let sample = 1; sample <= samples; sample++) {
				console.log(`[${source}] "${prompt}" sample ${sample}/${samples}`);
				const started = Date.now();
				try {
					const { response, sources } = await captureOpenAiResponse({
						prompt,
						model,
						grounded: mode === "grounded",
					});
					push({
						source,
						prompt,
						sample,
						response,
						sources,
						durationMs: Date.now() - started,
						attempts: 1,
					});
				} catch (err) {
					console.error(`[${source}] failed: ${errMessage(err)}`);
					push({
						source,
						prompt,
						sample,
						response: "",
						sources: [],
						error: errMessage(err),
						durationMs: Date.now() - started,
						attempts: 1,
						errorType: classifyError(err),
					});
				}
			}
		}
	}
}

async function main(): Promise<void> {
	if (process.argv[2] === "--recompute") {
		const jsonArg = process.argv[3];
		if (!jsonArg) throw new Error("--recompute requires a path to a run JSON");
		await recompute(path.resolve(process.cwd(), jsonArg));
		return;
	}

	if (process.argv[2] === "--analyze") {
		const paths = process.argv.slice(3).filter((a) => !a.startsWith("--"));
		if (paths.length === 0)
			throw new Error("--analyze requires at least one run JSON path");
		await analyze(paths.map((p) => path.resolve(process.cwd(), p)));
		return;
	}

	if (process.argv[2] === "--export") {
		const jsonArg = process.argv[3];
		if (!jsonArg) throw new Error("--export requires a path to a run JSON");
		exportRun(path.resolve(process.cwd(), jsonArg));
		return;
	}

	const config = loadConfig();
	const model = config.model ?? process.env.AZURE_OPENAI_DEPLOYMENT;
	if (!model) {
		throw new Error(
			"No model/deployment set. Add `model` to the config or set AZURE_OPENAI_DEPLOYMENT in .env.",
		);
	}
	const prompts = normalizePrompts(config);
	const brandDict = resolveBrandDict(config);
	console.log(
		`Comparing ${prompts.length} prompt(s) × ${config.samples} sample(s) — ` +
			`web + api ${config.modes.join("/")} (model ${model})`,
	);

	const capturesByPrompt = new Map<string, CapturedResponse[]>();
	for (const p of prompts) capturesByPrompt.set(p.text, []);
	const push = (capture: CapturedResponse) => {
		capturesByPrompt.get(capture.prompt)?.push(capture);
	};

	const outDir = path.resolve(here, "..", "compare-output");
	const failuresDir = path.join(outDir, "failures");
	fs.mkdirSync(failuresDir, { recursive: true });

	const webMeta: WebSampleMeta[] = [];
	await captureWeb(prompts, config.samples, failuresDir, push, webMeta);
	await captureApi(prompts, config.samples, config.modes, model, push);

	const results: PromptResult[] = prompts.map((p) => ({
		prompt: p.text,
		captures: capturesByPrompt.get(p.text) ?? [],
		intent: p.intent,
		brand: p.brand,
	}));

	console.log("Building report…");
	// Legacy report uses naive substring brand counts (denominator canonicals);
	// the authoritative alias/canonical SoV lives in the --analyze output.
	const { markdown, json, csv } = await buildComparisonReport(results, {
		brands: denominatorBrands(brandDict),
		model,
		samples: config.samples,
		modes: config.modes,
		embeddings: config.embeddings,
	});

	// Merge per-web-sample diagnostics + raw captures + brand dict into the
	// report JSON. `results` (with intent/brand) + `brandDict` let the run be
	// re-analyzed later (`--analyze`/`--recompute`) with no new captures.
	const reportObj = JSON.parse(json);
	reportObj.webSamples = webMeta;
	reportObj.results = results;
	reportObj.brandDict = brandDict;

	const stamp = timestamp();
	writeReport(outDir, `report-${stamp}`, markdown, reportObj, csv);

	// Response archive on every run (the RFP's response-archive requirement).
	const transcriptPath = path.join(outDir, `transcript-${stamp}.md`);
	const responsesPath = path.join(outDir, `responses-${stamp}.csv`);
	fs.writeFileSync(transcriptPath, buildTranscript(results));
	fs.writeFileSync(responsesPath, buildResponsesCsv(results));
	console.log(`  ${transcriptPath}\n  ${responsesPath}`);
}

function timestamp(): string {
	return new Date().toISOString().replace(/[:.]/g, "-");
}

function writeReport(
	outDir: string,
	baseName: string,
	markdown: string,
	reportObj: unknown,
	csv: string,
): string {
	fs.mkdirSync(outDir, { recursive: true });
	const base = path.join(outDir, baseName);
	fs.writeFileSync(`${base}.md`, markdown);
	fs.writeFileSync(`${base}.json`, JSON.stringify(reportObj, null, 2));
	fs.writeFileSync(`${base}.csv`, csv);
	console.log(
		`\nDone. Report written to:\n  ${base}.md\n  ${base}.json\n  ${base}.csv`,
	);
	return base;
}

/**
 * Regenerates a report from an existing run's captured responses (no new
 * captures), with embeddings forced on. Requires the run JSON to contain the
 * raw `results` — reports written before capture persistence won't have them.
 */
async function recompute(jsonPath: string): Promise<void> {
	const data = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
	const results: PromptResult[] | undefined = data.results;
	if (!Array.isArray(results) || results.length === 0) {
		throw new Error(
			`${path.basename(jsonPath)} has no raw captured responses (\`results\`) — it predates ` +
				"capture persistence, so embeddings can't be backfilled without re-capturing.",
		);
	}

	const opts: ReportOptions = { ...data.options, embeddings: true };
	console.log(
		`Recomputing ${results.length} prompt(s) from ${path.basename(jsonPath)} with embeddings on…`,
	);
	const { markdown, json, csv } = await buildComparisonReport(results, opts);

	const reportObj = JSON.parse(json);
	reportObj.results = results;
	if (data.webSamples) reportObj.webSamples = data.webSamples;

	const outDir = path.resolve(here, "..", "compare-output");
	writeReport(outDir, `recompute-${timestamp()}`, markdown, reportObj, csv);
	console.log(`\n${markdown}`);
}

/** Loads a persisted run JSON into RunData; brand dict from the run if present. */
function loadRunFile(jsonPath: string): {
	run: RunData;
	brandDict?: BrandEntry[];
} {
	const data = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
	const results: PromptResult[] | undefined = data.results;
	if (!Array.isArray(results) || results.length === 0) {
		throw new Error(
			`${path.basename(jsonPath)} has no raw captured responses (\`results\`) — it ` +
				"predates capture persistence, so it can't be analyzed without re-capturing.",
		);
	}
	return {
		run: { options: data.options, results, webSamples: data.webSamples },
		brandDict: data.brandDict,
	};
}

/**
 * Worth-it evaluation from one or more persisted runs (no new captures). The
 * brand dictionary comes from the run JSON when present, else is synthesized
 * from the legacy `options.brands` as competitors. Writes analysis-<stamp>.*.
 */
async function analyze(jsonPaths: string[]): Promise<void> {
	const loaded = jsonPaths.map(loadRunFile);
	const runs = loaded.map((l) => l.run);
	const brandDict: BrandEntry[] =
		loaded.find((l) => l.brandDict?.length)?.brandDict ??
		(runs[0]?.options.brands ?? []).map((b) => ({
			canonical: b,
			aliases: [b],
			role: "competitor" as const,
		}));
	console.log(
		`Analyzing ${runs.length} run(s), ${brandDict.length} brand(s) in dictionary…`,
	);
	const { markdown, json, csv } = await buildEvaluationAnalysis(runs, {
		brandDict,
	});

	const outDir = path.resolve(here, "..", "compare-output");
	fs.mkdirSync(outDir, { recursive: true });
	const base = path.join(outDir, `analysis-${timestamp()}`);
	fs.writeFileSync(`${base}.md`, markdown);
	fs.writeFileSync(`${base}.json`, json);
	fs.writeFileSync(`${base}.csv`, csv);
	console.log(
		`\nDone. Analysis written to:\n  ${base}.md\n  ${base}.json\n  ${base}.csv`,
	);
	console.log(`\n${markdown}`);
}

/** Writes the transcript + full-text responses CSV for a persisted run. */
function exportRun(jsonPath: string): void {
	const { run } = loadRunFile(jsonPath);
	const outDir = path.resolve(here, "..", "compare-output");
	fs.mkdirSync(outDir, { recursive: true });
	const stamp = timestamp();
	const transcriptPath = path.join(outDir, `transcript-${stamp}.md`);
	const responsesPath = path.join(outDir, `responses-${stamp}.csv`);
	fs.writeFileSync(transcriptPath, buildTranscript(run.results));
	fs.writeFileSync(responsesPath, buildResponsesCsv(run.results));
	console.log(
		`\nDone. Exports written to:\n  ${transcriptPath}\n  ${responsesPath}`,
	);
}

// Only run the full harness when invoked directly (so the module can be
// imported — e.g. by a validation script — without side effects).
const invokedDirectly =
	process.argv[1] != null &&
	import.meta.url === pathToFileURL(process.argv[1]).href;

if (invokedDirectly) {
	main().catch((err) => {
		console.error("Fatal:", err);
		process.exit(1);
	});
}
