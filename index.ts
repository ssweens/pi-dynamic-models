/**
 * pi-dynamic-models — Dynamic model discovery for Pi coding agent
 *
 * Reads ~/.pi/agent/settings/pi-dynamic-models.json and registers each configured
 * server as a named provider. Catalog discovery starts after session start and runs in the background.
 *
 * Config file (~/.pi/agent/settings/pi-dynamic-models.json):
 *
 *   [
 *     {
 *       "provider": "local-llm",
 *       "baseUrl": "http://192.168.1.51:9999/v1",
 *       "apiKey": "MY_API_KEY",
 *       "api": "openai-completions",
 *       "compat": {
 *         "supportsUsageInStreaming": true,
 *         "maxTokensField": "max_tokens"
 *       },
 *       "imageApi": "openrouter-images",
 *       "models": {
 *         "jev-model": {
 *           "architecture": {
 *             "input_modalities": ["text"],
 *             "output_modalities": ["decisions"]
 *           }
 *         }
 *       }
 *     }
 *   ]
 *
 * Fields:
 *   provider  (required) Name shown in the model selector
 *   baseUrl   (required) Server URL including /v1 if needed
 *   apiKey    (optional) Literal key, env var name, or !shell-command
 *   api       (optional) Chat API type; defaults to "openai-completions"
 *   imageApi  (optional) "openai-images" for /images/generations, "openrouter-images" for chat image output
 *   compat    (optional) OpenAI-completions compat overrides (see OpenAICompat)
 *   models    (optional) Per-model metadata keyed by model ID; overrides discovered metadata
 *
 * Discovery + registration logic:
 *   1. Read OpenRouter-style architecture modalities from discovered model records.
 *   2. OpenRouter defaults fetch the text, image, and decision listings and merge by model ID.
 *   3. Register text as chat, image through its configured API, and decisions as System One classifiers.
 *      Unsupported output types are never silently registered as chat.
 *
 * Explicit models register before discovery. Unreachable sources leave those
 * fallbacks available; providers without configured models are skipped.
 */

import type { AnyModel } from "@earendil-works/pi-ai";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI, ExtensionContext, ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { exec } from "node:child_process";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { openAIImagesApi } from "./image-api.js";

type DynamicImageApi = "openrouter-images" | "openai-images";

// Locate the active pi-ai package across managed releases, global installs, and Jiti's CJS resolver.
function resolvePiAiDirectory(): string {
  for (const installRoot of [
    process.env.PI_MANAGED_INSTALL_ROOT,
    join(getAgentDir(), "install"),
    join(homedir(), ".pi", "agent", "install"),
  ]) {
    const versionFile = installRoot && join(installRoot, "current-version");
    if (!installRoot || !versionFile || !existsSync(versionFile)) continue;
    const version = readFileSync(versionFile, "utf8").trim();
    const packageRoot = join(installRoot, "releases", version, "node_modules", "@earendil-works", "pi-ai");
    if (existsSync(join(packageRoot, "package.json"))) return packageRoot;
  }

  if (process.argv[1]) {
    try {
      let dir = dirname(realpathSync(process.argv[1]));
      for (;;) {
        for (const candidate of [join(dir, "node_modules", "@earendil-works", "pi-ai"), join(dir, "ai")]) {
          if (existsSync(join(candidate, "package.json"))) return candidate;
        }
        const parent = dirname(dir);
        if (parent === dir) break;
        dir = parent;
      }
    } catch {}
  }

  try {
    return dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai")));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const match = message.match(/No "exports" main defined in (.+)[\\/]package\.json/);
    if (match && existsSync(join(match[1], "package.json"))) return match[1];
    throw error;
  }
}

let aiDirectory = resolvePiAiDirectory();
let aiPackage: { name?: string; exports?: Record<string, { import: string }> };
for (;;) {
  const manifest = join(aiDirectory, "package.json");
  aiPackage = existsSync(manifest) ? JSON.parse(readFileSync(manifest, "utf8")) : {};
  if (aiPackage.name === "@earendil-works/pi-ai") break;
  const parent = dirname(aiDirectory);
  if (parent === aiDirectory) throw new Error("Cannot resolve pi-ai provider API exports.");
  aiDirectory = parent;
}
const apiExport = aiPackage.exports!["./api/*"].import;
const apiUrl = (name: string) => new URL(apiExport.replace("*", name), pathToFileURL(join(aiDirectory, "package.json")));
const loadOptionalApi = async (name: string) => {
  const url = apiUrl(name);
  return existsSync(fileURLToPath(url)) ? import(url.href) : {};
};
const { openrouterImagesApi } = await loadOptionalApi("openrouter-images.lazy");
const { typesafeSystemOneApi } = await loadOptionalApi("typesafe-system-one.lazy");

function truncatePlain(text: string, width: number): string {
  if (width <= 0) return "";
  if (text.length <= width) return text;
  if (width === 1) return "…";
  return `${text.slice(0, width - 1)}…`;
}

const execAsync = promisify(exec);

function resolveStaticApiKey(apiKey?: string): string | undefined {
  if (!apiKey || apiKey === "none" || apiKey.startsWith("!")) return undefined;

  const templateMatch = apiKey.match(/^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$/);
  if (templateMatch) {
    const envName = templateMatch[1] ?? templateMatch[2];
    return process.env[envName] || undefined;
  }

  return process.env[apiKey] ?? apiKey;
}

async function resolveConfiguredApiKey(apiKey?: string, signal?: AbortSignal): Promise<string | undefined> {
  if (!apiKey || apiKey === "none") return undefined;

  if (apiKey.startsWith("!")) {
    try {
      const { stdout } = await execAsync(apiKey.slice(1), {
        encoding: "utf8",
        timeout: 10_000,
        signal,
      });
      return stdout.trim() || undefined;
    } catch {
      return undefined;
    }
  }

  return resolveStaticApiKey(apiKey);
}

function requestSignal(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  return signal
    ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)])
    : AbortSignal.timeout(timeoutMs);
}

// Mirrors OpenAICompletionsCompat from @earendil-works/pi-ai
interface OpenAICompat {
  supportsStore?: boolean;
  supportsDeveloperRole?: boolean;
  supportsReasoningEffort?: boolean;
  supportsUsageInStreaming?: boolean;
  maxTokensField?: "max_completion_tokens" | "max_tokens";
  requiresToolResultName?: boolean;
  requiresAssistantAfterToolResult?: boolean;
  requiresThinkingAsText?: boolean;
  requiresMistralToolIds?: boolean;
  thinkingFormat?: "openai" | "zai" | "qwen";
  supportsStrictMode?: boolean;
}

interface ModelArchitecture {
  input_modalities?: string[];
  output_modalities?: string[];
}

interface ModelOverride {
  name?: string;
  reasoning?: boolean;
  input?: ("text" | "image")[];
  contextWindow?: number;
  maxTokens?: number;
  architecture?: ModelArchitecture;
  imageApi?: DynamicImageApi;
}

interface ModelSourceConfig {
  url: string;
  itemsPath?: string;
  idPath?: string;
  namePath?: string;
}

interface ServerConfig {
  provider: string;
  baseUrl: string;
  apiKey?: string;
  api?: string;
  compat?: OpenAICompat;
  imageApi?: DynamicImageApi;
  modelsSource?: ModelSourceConfig;
  models?: Record<string, ModelOverride>;
}

interface DiscoveredModel {
  id: string;
  name?: string;
  architecture?: ModelArchitecture;
}

/** Shape returned by GET {baseUrl}/corral/models */
interface CorralModelDetail {
  id: string;
  context_size: number | null;
  architecture?: ModelArchitecture;
  hf_base?: string | null;
  aliases?: string[];
  unlisted?: boolean;
  ttl?: number | null;
  pool_size?: number;
}

interface CorralModelsResponse {
  object: string;
  data: CorralModelDetail[];
}

function parseConfigFile(path: string): ServerConfig[] {
  if (!existsSync(path)) return [];

  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(path, "utf-8"));
  } catch (err) {
    console.warn(`[pi-dynamic-models] Failed to parse ${path}: ${err}`);
    return [];
  }

  if (!Array.isArray(parsed)) {
    console.warn(`[pi-dynamic-models] ${path} must be a JSON array`);
    return [];
  }

  const valid: ServerConfig[] = [];
  for (const entry of parsed) {
    if (!entry.provider || !entry.baseUrl) {
      console.warn(`[pi-dynamic-models] Skipping entry missing "provider" or "baseUrl": ${JSON.stringify(entry)}`);
      continue;
    }
    valid.push(entry as ServerConfig);
  }
  return valid;
}

/** Load and merge global + project configs.
 *  Project servers override global ones by provider name; extras are appended. */
function loadConfig(): ServerConfig[] {
  const configPath = join(getAgentDir(), "settings", "pi-dynamic-models.json");
  return parseConfigFile(configPath);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function getValueAtPath(value: unknown, path?: string): unknown {
  const normalizedPath = path?.trim();
  if (!normalizedPath || normalizedPath === ".") return value;

  let current: unknown = value;
  for (const part of normalizedPath.split(".").filter(Boolean)) {
    if (current === null || current === undefined) return undefined;
    if (Array.isArray(current) && /^\d+$/.test(part)) {
      current = current[Number(part)];
      continue;
    }
    if (!isRecord(current)) return undefined;
    current = current[part];
  }
  return current;
}

function toOptionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function readItemsAtPath(value: unknown, path?: string): unknown[] {
  const scoped = getValueAtPath(value, path);
  if (Array.isArray(scoped)) return scoped;
  if (!path && isRecord(value) && Array.isArray(value.data)) return value.data;
  return [];
}

function readModalities(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return Array.from(new Set(value.flatMap((entry) => {
    if (typeof entry !== "string") return [];
    const modality = entry.trim().toLowerCase();
    return /^[a-z][a-z0-9_-]*$/u.test(modality) ? [modality] : [];
  })));
}

function readArchitecture(value: unknown): ModelArchitecture | undefined {
  if (!isRecord(value)) return undefined;
  const inputModalities = readModalities(value.input_modalities);
  const outputModalities = readModalities(value.output_modalities);
  if (inputModalities === undefined && outputModalities === undefined) return undefined;
  return {
    ...(inputModalities !== undefined ? { input_modalities: inputModalities } : {}),
    ...(outputModalities !== undefined ? { output_modalities: outputModalities } : {}),
  };
}

function mergeModalities(left?: string[], right?: string[]): string[] | undefined {
  if (left === undefined) return right;
  if (right === undefined) return left;
  return Array.from(new Set([...left, ...right]));
}

export function mergeDiscoveredModels(modelLists: readonly DiscoveredModel[][]): DiscoveredModel[] {
  const merged = new Map<string, DiscoveredModel>();
  for (const models of modelLists) {
    for (const model of models) {
      const previous = merged.get(model.id);
      if (!previous) {
        merged.set(model.id, model);
        continue;
      }
      const inputModalities = mergeModalities(
        previous.architecture?.input_modalities,
        model.architecture?.input_modalities,
      );
      const outputModalities = mergeModalities(
        previous.architecture?.output_modalities,
        model.architecture?.output_modalities,
      );
      merged.set(model.id, {
        id: model.id,
        name: previous.name && previous.name !== previous.id ? previous.name : model.name ?? previous.name,
        ...(inputModalities !== undefined || outputModalities !== undefined
          ? {
            architecture: {
              ...(inputModalities !== undefined ? { input_modalities: inputModalities } : {}),
              ...(outputModalities !== undefined ? { output_modalities: outputModalities } : {}),
            },
          }
          : {}),
      });
    }
  }
  return Array.from(merged.values());
}

export function buildModelSources(baseUrl: string, configured?: ModelSourceConfig): ModelSourceConfig[] {
  if (configured?.url?.trim()) return [configured];
  const modelsUrl = `${baseUrl.replace(/\/+$/u, "")}/models`;
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(modelsUrl);
  } catch {
    return [{ url: modelsUrl }];
  }
  if (parsedUrl.hostname !== "openrouter.ai") return [{ url: modelsUrl }];

  return [undefined, "image", "decisions"].map((modality) => {
    const url = new URL(modelsUrl);
    if (modality) url.searchParams.set("output_modalities", modality);
    return { url: url.toString() };
  });
}

async function fetchCorralModelDetails(
  baseUrl: string,
  apiKey?: string,
  signal?: AbortSignal,
  fetcher: typeof fetch = fetch,
): Promise<Map<string, CorralModelDetail>> {
  const normalized = baseUrl.replace(/\/+$/, "");
  const rootCandidate = normalized.replace(/\/v\d+(?:\.\d+)?$/, "");
  const candidateUrls = Array.from(new Set([
    `${rootCandidate}/corral/models`,
    `${normalized}/corral/models`,
  ]));

  const headers: Record<string, string> = { Accept: "application/json" };
  if (apiKey && apiKey !== "none") headers["Authorization"] = `Bearer ${apiKey}`;

  for (const url of candidateUrls) {
    try {
      const response = await fetcher(url, { headers, signal: requestSignal(signal, 5000) });
      if (!response.ok) continue;
      const body = (await response.json()) as CorralModelsResponse;
      const map = new Map<string, CorralModelDetail>();
      for (const m of body.data ?? []) map.set(m.id, m);
      return map;
    } catch {
      if (signal?.aborted) return new Map();
      // try next candidate URL
    }
  }

  return new Map();
}

async function fetchModelsFromSource(
  source: ModelSourceConfig,
  apiKey?: string,
  fetcher: typeof fetch = fetch,
  signal?: AbortSignal,
): Promise<DiscoveredModel[]> {
  const url = source.url.trim();
  if (!url) {
    throw new Error("model source url is required");
  }

  const headers: Record<string, string> = { Accept: "application/json" };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  const response = await fetcher(url, {
    headers,
    signal: requestSignal(signal, 8000),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }

  const body: unknown = await response.json();
  const items = readItemsAtPath(body, source.itemsPath);
  const idPath = source.idPath?.trim() || "id";
  const namePath = source.namePath?.trim() || "name";

  return items.flatMap((item) => {
    if (!isRecord(item)) return [];
    const id = toOptionalString(getValueAtPath(item, idPath))?.trim();
    if (!id) return [];
    const architecture = readArchitecture(item.architecture);
    return [{
      id,
      name: toOptionalString(getValueAtPath(item, namePath))?.trim() || id,
      ...(architecture ? { architecture } : {}),
    }];
  });
}

export async function fetchModelsFromSources(
  sources: ModelSourceConfig[],
  apiKey?: string,
  fetcher: typeof fetch = fetch,
  signal?: AbortSignal,
): Promise<{ models: DiscoveredModel[]; warnings: string[] }> {
  const results = await Promise.all(sources.map(async (source) => {
    try {
      return { source, models: await fetchModelsFromSource(source, apiKey, fetcher, signal) };
    } catch (error) {
      return { source, error };
    }
  }));
  const successful = results.filter((result): result is { source: ModelSourceConfig; models: DiscoveredModel[] } =>
    "models" in result
  );
  const failed = results.filter((result): result is { source: ModelSourceConfig; error: unknown } =>
    "error" in result
  );
  if (successful.length === 0) {
    const failures = failed.map(({ source, error }) => `${source.url} (${error})`).join("; ");
    throw new Error(failures || "no model catalog sources configured");
  }
  return {
    models: mergeDiscoveredModels(successful.map((result) => result.models)),
    warnings: failed.map(({ source, error }) => `Could not reach ${source.url} (${error})`),
  };
}

function resolveArchitecture(
  discovered: DiscoveredModel | undefined,
  override: ModelOverride | undefined,
): ModelArchitecture | undefined {
  const discoveredArchitecture = readArchitecture(discovered?.architecture);
  const overrideArchitecture = readArchitecture(override?.architecture);
  const inputModalities = overrideArchitecture?.input_modalities
    ?? discoveredArchitecture?.input_modalities;
  const outputModalities = overrideArchitecture?.output_modalities
    ?? discoveredArchitecture?.output_modalities;
  if (inputModalities === undefined && outputModalities === undefined) return undefined;
  return {
    ...(inputModalities !== undefined ? { input_modalities: inputModalities } : {}),
    ...(outputModalities !== undefined ? { output_modalities: outputModalities } : {}),
  };
}

function supportedInputs(values: unknown): ("text" | "image")[] {
  if (values === undefined) return ["text"];
  if (!Array.isArray(values)) return [];
  return Array.from(new Set(values.filter((value): value is "text" | "image" =>
    value === "text" || value === "image"
  )));
}

export function resolveImageApi(baseUrl: string, configured?: string): DynamicImageApi | undefined {
  if (configured === "openrouter-images" || configured === "openai-images") return configured;
  try {
    if (new URL(baseUrl).hostname === "openrouter.ai") return "openrouter-images";
  } catch {
    // Invalid endpoints retain the chat-only fallback.
  }
  return undefined;
}

export function buildModelEntries(options: {
  ids: string[];
  provider: string;
  baseUrl: string;
  api: string;
  imageApi?: string;
  compat?: OpenAICompat;
  discoveredById: ReadonlyMap<string, DiscoveredModel>;
  overrides?: Record<string, ModelOverride>;
  corralDetails: ReadonlyMap<string, CorralModelDetail>;
}): {
  models: AnyModel[];
  unsupportedModels: number;
  skippedImageModels: number;
  unsupportedInputModalities: string[];
  unsupportedOutputModalities: string[];
} {
  const models: AnyModel[] = [];
  const unsupportedInputs = new Set<string>();
  const unsupportedOutputs = new Set<string>();
  let unsupportedModels = 0;
  let skippedImageModels = 0;

  for (const id of options.ids) {
    const override = options.overrides?.[id];
    const discovered = options.discoveredById.get(id);
    const architecture = resolveArchitecture({
      id,
      architecture: { ...options.corralDetails.get(id)?.architecture, ...discovered?.architecture },
    }, override);
    for (const modality of architecture?.input_modalities ?? []) {
      if (modality !== "text" && modality !== "image") unsupportedInputs.add(modality);
    }
    const outputs = architecture?.output_modalities ?? ["text"];
    for (const modality of outputs) {
      if (modality !== "text" && modality !== "image" && modality !== "decisions") {
        unsupportedOutputs.add(modality);
      }
    }
    const inputs = supportedInputs(override?.input ?? architecture?.input_modalities);
    const contextWindow = override?.contextWindow
      ?? options.corralDetails.get(id)?.context_size
      ?? 128_000;
    const common = {
      id,
      name: override?.name ?? discovered?.name ?? id,
      provider: options.provider,
      baseUrl: options.baseUrl,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    };
    let registered = false;

    if (outputs.includes("text") && inputs.length > 0) {
      models.push({
        ...common,
        type: "chat",
        api: options.api,
        reasoning: override?.reasoning ?? false,
        input: inputs,
        contextWindow,
        maxTokens: override?.maxTokens ?? 16_384,
        ...(options.compat ? { compat: options.compat } : {}),
      });
      registered = true;
    }

    if (outputs.includes("image")) {
      const imageApi = override?.imageApi ?? options.imageApi;
      if ((imageApi === "openrouter-images" || imageApi === "openai-images") && inputs.includes("text")) {
        models.push({
          ...common,
          type: "image",
          api: imageApi,
          input: imageApi === "openai-images" ? ["text"] : inputs,
          output: outputs.filter((value): value is "text" | "image" =>
            value === "text" || value === "image"
          ),
        });
        registered = true;
      } else {
        skippedImageModels++;
      }
    }

    if (outputs.includes("decisions")) {
      if (inputs.includes("text")) {
        models.push({
          ...common,
          type: "classifier",
          api: "typesafe-system-one",
          input: ["text"],
          contextWindow,
        });
        registered = true;
      }
    }

    if (!registered) unsupportedModels++;
  }

  return {
    models,
    unsupportedModels,
    skippedImageModels,
    unsupportedInputModalities: Array.from(unsupportedInputs).sort(),
    unsupportedOutputModalities: Array.from(unsupportedOutputs).sort(),
  };
}

export function buildOperationHandlers(models: AnyModel[]) {
  return {
    ...(models.some((model) => model.type === "image") ? {
      images: {
        ...(openrouterImagesApi && models.some((model) => model.type === "image" && model.api === "openrouter-images")
          ? { "openrouter-images": openrouterImagesApi() } : {}),
        ...(models.some((model) => model.type === "image" && model.api === "openai-images")
          ? { "openai-images": openAIImagesApi() } : {}),
      },
    } : {}),
    ...(typesafeSystemOneApi && models.some((model) => model.type === "classifier")
      ? { classifiers: { "typesafe-system-one": typesafeSystemOneApi() } }
      : {}),
  };
}

interface StartupSummary {
  unavailableSources: number;
  configuredFallbackProviders: number;
  skippedProviders: number;
  emptyProviders: number;
  unsupportedModels: number;
  skippedImageModels: number;
}

export function formatStartupSummary(summary: StartupSummary): string[] {
  const lines: string[] = [];
  const discovery: string[] = [];
  if (summary.unavailableSources > 0) {
    discovery.push(`unavailable model sources: ${summary.unavailableSources}`);
    if (summary.configuredFallbackProviders > 0) {
      discovery.push(`configured fallback providers: ${summary.configuredFallbackProviders}`);
    }
    if (summary.skippedProviders > 0) discovery.push(`skipped providers: ${summary.skippedProviders}`);
  }
  if (discovery.length > 0) lines.push(`[pi-dynamic-models] ${discovery.join("; ")}`);

  if (summary.emptyProviders > 0) {
    lines.push(`[pi-dynamic-models] Providers with no models: ${summary.emptyProviders}`);
  }
  const unsupported: string[] = [];
  if (summary.unsupportedModels > 0) unsupported.push(`unsupported models: ${summary.unsupportedModels}`);
  if (summary.skippedImageModels > 0) {
    unsupported.push(`images requiring a supported imageApi: ${summary.skippedImageModels}`);
  }
  if (unsupported.length > 0) lines.push(`[pi-dynamic-models] ${unsupported.join("; ")}`);
  return lines;
}

function createStartupSummary(): StartupSummary {
  return {
    unavailableSources: 0,
    configuredFallbackProviders: 0,
    skippedProviders: 0,
    emptyProviders: 0,
    unsupportedModels: 0,
    skippedImageModels: 0,
  };
}

interface DiscoveryRun {
  controller: AbortController;
  ui: ExtensionUIContext;
  dismissed: boolean;
  summary: StartupSummary;
}

function updateStartupWidget(run: DiscoveryRun): void {
  const lines = formatStartupSummary(run.summary);
  run.ui.setWidget(
    "pi-dynamic-models-startup",
    !run.dismissed && lines.length > 0
      ? (_tui, theme) => ({
        render: (width: number) => [
          ...lines.map((line) => theme.fg("muted", truncatePlain(line, width))),
          "",
        ],
        invalidate: () => {},
      })
      : undefined,
  );
}

function registerProviderModels(
  pi: ExtensionAPI,
  server: ServerConfig,
  apiKey: string | undefined,
  fetchedModels: DiscoveredModel[],
  corralDetails: ReadonlyMap<string, CorralModelDetail>,
) {
  const normalizedBaseUrl = server.baseUrl.replace(/\/+$/u, "");
  const modelIds = Array.from(new Set([
    ...fetchedModels.map((model) => model.id),
    ...Object.keys(server.models ?? {}),
  ])).sort((a, b) => a.localeCompare(b));
  const registration = buildModelEntries({
    ids: modelIds,
    provider: server.provider,
    baseUrl: normalizedBaseUrl,
    api: server.api ?? "openai-completions",
    imageApi: resolveImageApi(normalizedBaseUrl, server.imageApi),
    compat: server.compat,
    discoveredById: new Map(fetchedModels.map((model) => [model.id, model])),
    overrides: server.models,
    corralDetails,
  });

  if (registration.models.length > 0) {
    pi.registerProvider(server.provider, {
      baseUrl: normalizedBaseUrl,
      // Supports literal keys, env var names, $ENV_VAR references, and !shell-commands.
      apiKey: apiKey ?? server.apiKey ?? "none",
      authHeader: Boolean(apiKey),
      api: server.api ?? "openai-completions",
      models: registration.models,
      ...buildOperationHandlers(registration.models),
    });
  }
  return registration;
}

async function hydrateProvider(pi: ExtensionAPI, server: ServerConfig, run: DiscoveryRun): Promise<void> {
  const { signal } = run.controller;
  const resolvedApiKey = await resolveConfiguredApiKey(server.apiKey, signal);
  if (signal.aborted) return;

  // Shell-based keys need an async process, so register their fallback models once it resolves.
  if (server.apiKey?.startsWith("!") && Object.keys(server.models ?? {}).length > 0) {
    registerProviderModels(pi, server, resolvedApiKey, [], new Map());
  }

  const normalizedBaseUrl = server.baseUrl.replace(/\/+$/u, "");
  const sources = buildModelSources(normalizedBaseUrl, server.modelsSource);
  let fetchedModels: DiscoveredModel[] = [];
  try {
    const discovery = await fetchModelsFromSources(sources, resolvedApiKey, fetch, signal);
    fetchedModels = discovery.models;
    run.summary.unavailableSources += discovery.warnings.length;
  } catch {
    if (signal.aborted) return;
    run.summary.unavailableSources += sources.length;
    if (Object.keys(server.models ?? {}).length > 0) {
      run.summary.configuredFallbackProviders++;
    } else {
      run.summary.skippedProviders++;
      updateStartupWidget(run);
      return;
    }
  }

  if (signal.aborted) return;
  const modelIds = new Set([...fetchedModels.map((model) => model.id), ...Object.keys(server.models ?? {})]);
  if (modelIds.size === 0) {
    run.summary.emptyProviders++;
    updateStartupWidget(run);
    return;
  }

  // Publish each provider as soon as its catalog arrives; context sizes follow separately.
  const registration = registerProviderModels(pi, server, resolvedApiKey, fetchedModels, new Map());
  run.summary.unsupportedModels += registration.unsupportedModels;
  run.summary.skippedImageModels += registration.skippedImageModels;
  updateStartupWidget(run);
  if (registration.models.length === 0) return;

  const corralDetails = await fetchCorralModelDetails(server.baseUrl, resolvedApiKey, signal);
  if (signal.aborted || corralDetails.size === 0) return;
  registerProviderModels(pi, server, resolvedApiKey, fetchedModels, corralDetails);
}

/** Refresh operation-only catalogs without replacing native chat metadata or authentication. */
export async function hydrateOpenRouter(
  pi: ExtensionAPI,
  ctx: Pick<ExtensionContext, "modelRegistry">,
  signal?: AbortSignal,
): Promise<void> {
  const native = ctx.modelRegistry.getProvider("openrouter");
  const baseUrl = native?.baseUrl;
  if (!baseUrl || new URL(baseUrl).hostname !== "openrouter.ai") return;

  const discovery = await fetchModelsFromSources(buildModelSources(baseUrl).slice(1), undefined, fetch, signal);
  if (signal?.aborted) return;
  // Keep the existing operations on partial catalog failures rather than silently removing them.
  if (discovery.warnings.length) throw new Error(discovery.warnings.join("; "));

  const registration = buildModelEntries({
    ids: discovery.models.map((model) => model.id),
    provider: "openrouter", baseUrl, api: "openai-completions", imageApi: "openrouter-images",
    discoveredById: new Map(discovery.models.map((model) => [model.id, model])),
    corralDetails: new Map(),
  });
  const operations = registration.models.filter((model) => model.type === "image" || model.type === "classifier");
  pi.registerProvider({
    ...native,
    getModels: () => native.getModels(),
    getAllModels: () => {
      const current = native.getAllModels?.() ?? native.getModels();
      return [
        ...current.filter((model) => !model.type || model.type === "chat"),
        ...operations.map((model) => {
          const previous = current.find((old) => old.type === model.type && old.id === model.id);
          if (model.type === "image") return {
            ...model, ...(previous?.type === "image" ? previous : {}),
            api: model.api, input: model.input, output: model.output,
          };
          return {
            ...model, ...(previous?.type === "classifier" ? previous : {}),
            api: model.api, input: model.input,
          };
        }),
      ];
    },
  });
}

export default function (pi: ExtensionAPI): void {
  const servers = loadConfig();

  // Explicit models stay available immediately without waiting for any network calls.
  for (const server of servers) {
    if (server.apiKey?.startsWith("!")) continue;
    if (Object.keys(server.models ?? {}).length === 0) continue;
    registerProviderModels(pi, server, resolveStaticApiKey(server.apiKey), [], new Map());
  }

  let currentRun: DiscoveryRun | undefined;
  pi.on("session_start", (_event, ctx) => {
    currentRun?.controller.abort();
    const run: DiscoveryRun = {
      controller: new AbortController(),
      ui: ctx.ui,
      dismissed: false,
      summary: createStartupSummary(),
    };
    currentRun = run;
    updateStartupWidget(run);

    // Do not return or await this work: providers hydrate independently as discovery completes.
    if (!servers.some((server) => server.provider === "openrouter")) {
      void hydrateOpenRouter(pi, ctx, run.controller.signal).catch((error) => {
        if (!run.controller.signal.aborted) {
          run.summary.unavailableSources++;
          updateStartupWidget(run);
          console.error(`[pi-dynamic-models] OpenRouter discovery failed: ${error}`);
        }
      });
    }
    for (const server of servers) {
      void hydrateProvider(pi, server, run).catch((error) => {
        if (!run.controller.signal.aborted) {
          console.error(`[pi-dynamic-models] Failed to hydrate ${server.provider}: ${error}`);
        }
      });
    }
  });

  pi.on("session_shutdown", () => {
    currentRun?.controller.abort();
    currentRun = undefined;
  });

  pi.on("input", (_event, ctx) => {
    if (currentRun) currentRun.dismissed = true;
    ctx.ui.setWidget("pi-dynamic-models-startup", undefined);
  });
}
