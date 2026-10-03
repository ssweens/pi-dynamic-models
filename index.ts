/**
 * pi-dynamic-models — Dynamic model discovery for Pi coding agent
 *
 * Reads ~/.pi/agent/settings/pi-dynamic-models.json and registers each configured
 * server as a named provider by fetching a configurable model source at startup.
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
 *   imageApi  (optional) "openrouter-images" when the endpoint supports that wire format
 *   compat    (optional) OpenAI-completions compat overrides (see OpenAICompat)
 *   models    (optional) Per-model metadata keyed by model ID; overrides discovered metadata
 *
 * Discovery + registration logic:
 *   1. Read OpenRouter-style architecture modalities from discovered model records.
 *   2. OpenRouter defaults fetch the text, image, and decision listings and merge by model ID.
 *   3. Register text as chat, image through its configured API, and decisions as System One classifiers.
 *      Unsupported output types are never silently registered as chat.
 *
 * Servers that are unreachable at startup fall back to only the explicitly
 * configured models (if any). Servers with no configured models and an
 * unreachable /models endpoint are skipped entirely.
 */

import { openrouterImagesApi } from "@earendil-works/pi-ai/api/openrouter-images.lazy";
import { typesafeSystemOneApi } from "@earendil-works/pi-ai/api/typesafe-system-one.lazy";
import type { AnyModel } from "@earendil-works/pi-ai";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

function truncatePlain(text: string, width: number): string {
  if (width <= 0) return "";
  if (text.length <= width) return text;
  if (width === 1) return "…";
  return `${text.slice(0, width - 1)}…`;
}

function resolveConfiguredApiKey(apiKey?: string): string | undefined {
  if (!apiKey || apiKey === "none") return undefined;

  if (apiKey.startsWith("!")) {
    try {
      const resolved = execSync(apiKey.slice(1), {
        encoding: "utf-8",
        timeout: 10_000,
        stdio: ["ignore", "pipe", "ignore"],
      }).trim();
      return resolved || undefined;
    } catch {
      return undefined;
    }
  }

  const templateMatch = apiKey.match(/^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$/);
  if (templateMatch) {
    const envName = templateMatch[1] ?? templateMatch[2];
    return process.env[envName] || undefined;
  }

  return process.env[apiKey] ?? apiKey;
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
  imageApi?: "openrouter-images";
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
  imageApi?: "openrouter-images";
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
      const response = await fetch(url, { headers, signal: AbortSignal.timeout(5000) });
      if (!response.ok) continue;
      const body = (await response.json()) as CorralModelsResponse;
      const map = new Map<string, CorralModelDetail>();
      for (const m of body.data ?? []) map.set(m.id, m);
      return map;
    } catch {
      // try next candidate URL
    }
  }

  return new Map();
}

async function fetchModelsFromSource(
  source: ModelSourceConfig,
  apiKey?: string,
  fetcher: typeof fetch = fetch,
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
    signal: AbortSignal.timeout(8000),
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
): Promise<{ models: DiscoveredModel[]; warnings: string[] }> {
  const results = await Promise.all(sources.map(async (source) => {
    try {
      return { source, models: await fetchModelsFromSource(source, apiKey, fetcher) };
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

export function resolveImageApi(baseUrl: string, configured?: string): "openrouter-images" | undefined {
  if (configured === "openrouter-images") return configured;
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
    const architecture = resolveArchitecture(discovered, override);
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
      if (imageApi === "openrouter-images" && inputs.length > 0) {
        models.push({
          ...common,
          type: "image",
          api: imageApi,
          input: inputs,
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
    ...(models.some((model) => model.type === "image")
      ? { images: { "openrouter-images": openrouterImagesApi() } }
      : {}),
    ...(models.some((model) => model.type === "classifier")
      ? { classifiers: { "typesafe-system-one": typesafeSystemOneApi() } }
      : {}),
  };
}

export default async function (pi: ExtensionAPI): Promise<void> {
  const servers = loadConfig();
  if (servers.length === 0) return;

  const startupLines: string[] = [];

  await Promise.all(
    servers.map(async ({ provider, baseUrl, apiKey, api, compat, imageApi, modelsSource, models: modelOverrides }) => {
      const resolvedApiKey = resolveConfiguredApiKey(apiKey);
      const normalizedBaseUrl = baseUrl.replace(/\/+$/, "");
      const sources = buildModelSources(normalizedBaseUrl, modelsSource);
      const sourceLabel = sources.map((source) => source.url).join(", ");
      const registrationApiKey = resolvedApiKey ?? apiKey ?? "none";
      const useAuthHeader = Boolean(resolvedApiKey);

      // Fetch source metadata first; Corral context metadata remains additive.
      // If every source fetch fails but explicit models exist, keep those.
      let fetchedModels: DiscoveredModel[] = [];
      try {
        const discovery = await fetchModelsFromSources(sources, resolvedApiKey);
        fetchedModels = discovery.models;
        startupLines.push(...discovery.warnings.map((warning) => `   [pi-dynamic-models] ${warning}`));
      } catch (err) {
        if (modelOverrides && Object.keys(modelOverrides).length > 0) {
          startupLines.push(`   [pi-dynamic-models] Could not reach model sources ${sourceLabel} (${err}), using configured models only`);
        } else {
          startupLines.push(`   [pi-dynamic-models] Could not reach model sources ${sourceLabel}: ${err}`);
          return;
        }
      }

      let corralDetails = new Map<string, CorralModelDetail>();
      try {
        corralDetails = await fetchCorralModelDetails(baseUrl, resolvedApiKey);
      } catch {
        // Optional metadata only.
      }

      // Union of fetched IDs and explicitly configured IDs
      const allIds = new Set<string>([...fetchedModels.map((m) => m.id), ...Object.keys(modelOverrides ?? {})]);

      if (allIds.size === 0) {
        startupLines.push(`   [pi-dynamic-models] No models for provider "${provider}"`);
        return;
      }

      const discoveredById = new Map(fetchedModels.map((model) => [model.id, model]));
      const modelIds = Array.from(allIds).sort((a, b) => a.localeCompare(b));
      const registration = buildModelEntries({
        ids: modelIds,
        provider,
        baseUrl: normalizedBaseUrl,
        api: api ?? "openai-completions",
        imageApi: resolveImageApi(normalizedBaseUrl, imageApi),
        compat,
        discoveredById,
        overrides: modelOverrides,
        corralDetails,
      });

      if (registration.models.length === 0) {
        startupLines.push(`   [pi-dynamic-models] Provider "${provider}": no Pi-supported model operations (${allIds.size} IDs discovered/configured)`);
        if (registration.skippedImageModels > 0) {
          startupLines.push(`   [pi-dynamic-models] ${registration.skippedImageModels} image model(s) require imageApi "openrouter-images"`);
        }
        if (registration.unsupportedInputModalities.length > 0) {
          startupLines.push(`   [pi-dynamic-models] Unsupported input modalities not available to Pi: ${registration.unsupportedInputModalities.join(", ")}`);
        }
        if (registration.unsupportedOutputModalities.length > 0) {
          startupLines.push(`   [pi-dynamic-models] Unsupported output modalities not registered: ${registration.unsupportedOutputModalities.join(", ")}`);
        }
        return;
      }

      pi.registerProvider(provider, {
        baseUrl: normalizedBaseUrl,
        // Supports literal keys, env var names, $ENV_VAR references, and !shell-commands.
        apiKey: registrationApiKey,
        authHeader: useAuthHeader,
        api: api ?? "openai-completions",
        models: registration.models,
        ...buildOperationHandlers(registration.models),
      });

      const fetchedNote = fetchedModels.length > 0 ? `${fetchedModels.length} discovered` : "0 discovered";
      const overrideNote = Object.keys(modelOverrides ?? {}).length > 0
        ? `, ${Object.keys(modelOverrides!).length} configured`
        : "";
      const sourceNote = modelsSource?.url
        ? ` via ${modelsSource.url}`
        : sources.length > 1 ? " via /models + image/decisions listings" : " via /models";
      startupLines.push(
        `   [pi-dynamic-models] Provider "${provider}": ${fetchedNote}${overrideNote}, ${allIds.size} IDs / ${registration.models.length} operations (${api ?? "openai-completions"})${sourceNote}`
      );
      if (registration.unsupportedModels > 0) {
        startupLines.push(`   [pi-dynamic-models] ${registration.unsupportedModels} model(s) have no supported Pi operation; not registered as chat`);
      }
      if (registration.skippedImageModels > 0) {
        startupLines.push(`   [pi-dynamic-models] ${registration.skippedImageModels} image model(s) require imageApi "openrouter-images"`);
      }
      if (registration.unsupportedInputModalities.length > 0) {
        startupLines.push(`   [pi-dynamic-models] Unsupported input modalities not available to Pi: ${registration.unsupportedInputModalities.join(", ")}`);
      }
      if (registration.unsupportedOutputModalities.length > 0) {
        startupLines.push(`   [pi-dynamic-models] Unsupported output modalities not registered: ${registration.unsupportedOutputModalities.join(", ")}`);
      }
    })
  );

  // Show startup info as a widget that clears on first user input
  if (startupLines.length > 0) {
    pi.on("session_start", async (_event, ctx) => {
      ctx.ui.setWidget("pi-dynamic-models-startup", (_tui, theme) => ({
        render: (width: number) => [
          ...startupLines.map((line) => theme.fg("muted", truncatePlain(line, width))),
          "",
        ],
        invalidate: () => {},
      }));
    });
    pi.on("input", async (_event, ctx) => {
      ctx.ui.setWidget("pi-dynamic-models-startup", undefined);
    });
  }
}
