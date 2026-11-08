import { createModels, type AnyModel } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import dynamicModels, {
  buildModelEntries,
  buildModelSources,
  buildOperationHandlers,
  fetchModelsFromSources,
  formatStartupSummary,
  hydrateOpenRouter,
  resolveImageApi,
} from "./index.ts";

const emptyCorralDetails = () => new Map<string, { id: string; context_size: number | null }>();

type TestOverride = {
  name?: string;
  reasoning?: boolean;
  input?: ("text" | "image")[];
  contextWindow?: number;
  maxTokens?: number;
  architecture?: { input_modalities?: string[]; output_modalities?: string[] };
  imageApi?: "openrouter-images" | "openai-images";
};

function buildEntries(
  models: Array<{
    id: string;
    name?: string;
    contextWindow?: number;
    maxTokens?: number;
    architecture?: { input_modalities?: string[]; output_modalities?: string[] };
  }>,
  options: {
    imageApi?: string;
    overrides?: Record<string, TestOverride>;
    corralDetails?: Map<string, { id: string; context_size: number | null }>;
  } = {},
) {
  const discoveredById = new Map(models.map((model) => [model.id, model] as const));
  return buildModelEntries({
    ids: models.map((model) => model.id),
    provider: "corral-local",
    baseUrl: "http://corral.local/v1",
    api: "openai-completions",
    imageApi: options.imageApi,
    discoveredById,
    overrides: options.overrides,
    corralDetails: options.corralDetails ?? emptyCorralDetails(),
  });
}

describe("startup summary", () => {
  test("stays quiet when registration has no issues", () => {
    expect(formatStartupSummary({
      emptyProviders: 0,
      unsupportedModels: 0,
      skippedImageModels: 0,
    })).toEqual([]);
  });

  test("aggregates actionable startup issues into concise lines", () => {
    expect(formatStartupSummary({
      emptyProviders: 1,
      unsupportedModels: 3,
      skippedImageModels: 1,
    })).toEqual([
      '[pi-dynamic-models] Providers with no models: 1',
      '[pi-dynamic-models] unsupported models: 3; images requiring a supported imageApi: 1',
    ]);
  });
});

describe("background provider hydration", () => {
  test("returns from load immediately and registers each provider as discovery completes", async () => {
    const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
    const originalFetch = globalThis.fetch;
    const configDir = mkdtempSync(join(tmpdir(), "pi-dynamic-models-"));
    const handlers = new Map<string, (event: unknown, ctx: unknown) => unknown>();
    const registrations: Array<{ provider: string; models: AnyModel[] }> = [];
    let fetchCount = 0;
    let resolveFastHydrated!: () => void;
    const fastHydrated = new Promise<void>((resolve) => {
      resolveFastHydrated = resolve;
    });
    const fakePi = {
      on(event: string, handler: (event: unknown, ctx: unknown) => unknown) {
        handlers.set(event, handler);
        return () => {};
      },
      registerProvider(provider: string, config: { models: AnyModel[] }) {
        registrations.push({ provider, models: config.models });
        if (provider === "fast" && config.models.some((model) => model.id === "from-catalog")) {
          resolveFastHydrated();
        }
      },
    } as unknown as ExtensionAPI;

    const jsonResponse = (body: unknown) => new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });

    try {
      process.env.PI_CODING_AGENT_DIR = configDir;
      mkdirSync(join(configDir, "settings"), { recursive: true });
      writeFileSync(join(configDir, "settings", "pi-dynamic-models.json"), JSON.stringify([
        {
          provider: "fast",
          baseUrl: "http://fast.test/v1",
          apiKey: "none",
          models: { manual: { contextWindow: 64_000 } },
        },
        {
          provider: "slow",
          baseUrl: "http://slow.test/v1",
          apiKey: "none",
        },
        {
          provider: "offline",
          baseUrl: "https://openrouter.ai/api/v1",
          apiKey: "sk-or-test",
        },
      ]));
      globalThis.fetch = (async (input, init) => {
        fetchCount++;
        const url = new URL(String(input));
        if (url.hostname === "openrouter.ai") {
          // Unreachable the way offline DNS fails; must stay invisible to the user.
          throw new TypeError("fetch failed");
        }
        if (url.hostname === "slow.test") {
          return new Promise<Response>((_resolve, reject) => {
            const signal = init?.signal;
            if (signal?.aborted) reject(signal.reason);
            else signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
          });
        }
        if (url.pathname.endsWith("/corral/models")) return jsonResponse({ data: [] });
        return jsonResponse({ data: [{ id: "from-catalog" }] });
      }) as typeof fetch;

      expect(dynamicModels(fakePi)).toBeUndefined();
      expect(fetchCount).toBe(0);
      expect(registrations.map(({ provider }) => provider)).toEqual(["fast"]);
      expect(registrations[0].models.map((model) => model.id)).toEqual(["manual"]);

      const widgetStates: boolean[] = [];
      const ui = { setWidget: (_id: string, content: unknown) => { widgetStates.push(content !== undefined); } };
      const start = handlers.get("session_start");
      expect(start).toBeDefined();
      expect(start?.({ type: "session_start", reason: "startup" }, { ui, modelRegistry: { getProvider: () => undefined } })).toBeUndefined();
      expect(fetchCount).toBe(0);

      let timeout: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([
          fastHydrated,
          new Promise<never>((_resolve, reject) => {
            timeout = setTimeout(() => reject(new Error("fast provider did not hydrate")), 2_000);
          }),
        ]);
      } finally {
        if (timeout) clearTimeout(timeout);
      }

      const hydrated = registrations.find(({ provider, models }) =>
        provider === "fast" && models.some((model) => model.id === "from-catalog")
      );
      expect(hydrated?.models.map((model) => model.id)).toEqual(["from-catalog", "manual"]);
      expect(registrations.some(({ provider }) => provider === "slow")).toBe(false);

      // Unreachable OpenRouter must skip its provider silently: no registration, no widget content.
      await new Promise((resolve) => setTimeout(resolve, 250));
      expect(registrations.some(({ provider }) => provider === "offline")).toBe(false);
      expect(widgetStates.filter((hasContent) => hasContent)).toEqual([]);
    } finally {
      handlers.get("session_shutdown")?.({ type: "session_shutdown", reason: "quit" }, {});
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
      else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
      globalThis.fetch = originalFetch;
      rmSync(configDir, { recursive: true, force: true });
    }
  });
});

describe("OpenRouter modality discovery", () => {
  test("legacy catalogs keep registering as text chat", () => {
    const result = buildEntries([{ id: "legacy", name: "Legacy" }]);
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      type: "chat",
      id: "legacy",
      input: ["text"],
      api: "openai-completions",
    });
  });

  test("OpenRouter default discovery fetches chat, image, and decision catalogs", () => {
    const sources = buildModelSources("https://openrouter.ai/api/v1");
    expect(sources.map((source) => new URL(source.url).searchParams.get("output_modalities")))
      .toEqual([null, "image", "decisions"]);
    expect(buildModelSources("https://corral.local/v1")).toEqual([
      { url: "https://corral.local/v1/models" },
    ]);
  });

  test("an explicit model source is used without additional requests", () => {
    const configured = { url: "https://catalog.example/models?output_modalities=all", itemsPath: "data" };
    expect(buildModelSources("https://openrouter.ai/api/v1", configured)).toEqual([configured]);
  });

  test("merges modality listings and retains every raw catalog record", async () => {
    const fetcher: typeof fetch = async (input) => {
      const url = new URL(String(input));
      const output = url.searchParams.get("output_modalities") ?? "text";
      return new Response(JSON.stringify({
        data: [{
          id: "jev",
          name: "Jev",
          context_length: 262_144,
          top_provider: { context_length: 131_072, max_completion_tokens: 8_192 },
          pricing: {
            prompt: "0.00000008",
            completion: "0.0000002",
            input_cache_read: "0.00000002",
            input_cache_write: "0.0000003",
            request: "0.01",
            image: "0.02",
            overrides: [
              { min_prompt_tokens: 32_000, prompt: "0.00000004", completion: "0.0000001" },
              { utc_start: 12, prompt: "0.5" },
            ],
          },
          supported_parameters: ["tools"],
          architecture: {
            input_modalities: ["text"],
            output_modalities: [output],
          },
        }],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    };
    const result = await fetchModelsFromSources(
      buildModelSources("https://openrouter.ai/api/v1"),
      undefined,
      fetcher,
    );
    expect(result.warnings).toEqual([]);
    expect(result.models).toMatchObject([{
      id: "jev",
      name: "Jev",
      contextWindow: 262_144,
      maxTokens: 8_192,
      architecture: {
        input_modalities: ["text"],
        output_modalities: ["text", "image", "decisions"],
      },
    }]);
    expect(result.models[0].catalogMetadata).toHaveLength(3);
    expect(result.models[0].catalogMetadata?.[0]).toMatchObject({
      pricing: { prompt: "0.00000008", request: "0.01", image: "0.02" },
      supported_parameters: ["tools"],
    });
    const expectedCost = {
      input: 0.08,
      output: 0.2,
      cacheRead: 0.02,
      cacheWrite: 0.3,
      tiers: [{ inputTokensAbove: 32_000, input: 0.04, output: 0.1, cacheRead: 0.02, cacheWrite: 0.3 }],
    };
    expect(result.models[0].cost).toEqual(expectedCost);
    const registered = buildModelEntries({
      ids: ["jev"],
      provider: "openrouter",
      baseUrl: "https://openrouter.ai/api/v1",
      api: "openai-completions",
      discoveredById: new Map([["jev", result.models[0]]]),
      corralDetails: new Map(),
    });
    expect(registered.models[0].cost).toEqual(expectedCost);
  });

  test("does not assume other catalogs use OpenRouter token-price units", async () => {
    const result = await fetchModelsFromSources(
      [{ url: "https://catalog.example/models" }],
      undefined,
      async () => new Response(JSON.stringify({
        data: [{ id: "custom", pricing: { prompt: "0.00000008", completion: "0.0000002" } }],
      }), { status: 200, headers: { "Content-Type": "application/json" } }),
    );

    expect(result.models[0].cost).toBeUndefined();
    expect(result.models[0].catalogMetadata?.[0]).toMatchObject({
      pricing: { prompt: "0.00000008", completion: "0.0000002" },
    });
    const registered = buildModelEntries({
      ids: ["custom"],
      provider: "custom",
      baseUrl: "https://catalog.example",
      api: "openai-completions",
      discoveredById: new Map([["custom", result.models[0]]]),
      corralDetails: new Map(),
    });
    expect(registered.models[0].cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
  });

  test("normalizes OpenRouter and oMLX context and completion limits", async () => {
    const result = await fetchModelsFromSources(
      [
        { url: "https://openrouter.test/models" },
        { url: "https://omlx.test/v1/models" },
      ],
      undefined,
      async (input) => {
        const host = new URL(String(input)).hostname;
        const data = host === "openrouter.test"
          ? [{
            id: "or/model",
            context_length: 262_144,
            top_provider: { context_length: 131_072, max_completion_tokens: 8_192 },
          }]
          : [{ id: "local-model", max_model_len: 262_128, owned_by: "omlx" }];
        return new Response(JSON.stringify({ object: "list", data }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      },
    );

    expect(result.models).toMatchObject([
      { id: "or/model", contextWindow: 262_144, maxTokens: 8_192 },
      { id: "local-model", contextWindow: 262_128 },
    ]);
    expect(result.models.find((model) => model.id === "local-model")?.catalogMetadata?.[0])
      .toMatchObject({ owned_by: "omlx", max_model_len: 262_128 });
  });

  test("configured and Corral context limits precede catalog fallback", () => {
    const model = { id: "model", contextWindow: 262_144, maxTokens: 8_192 };
    const corralDetails = new Map([[
      "model",
      { id: "model", context_size: 131_072 },
    ]]);
    const corral = buildEntries([model], { corralDetails });
    expect(corral.models[0]).toMatchObject({ contextWindow: 131_072, maxTokens: 8_192 });

    const configured = buildEntries([model], {
      corralDetails,
      overrides: { model: { contextWindow: 65_536, maxTokens: 4_096 } },
    });
    expect(configured.models[0]).toMatchObject({ contextWindow: 65_536, maxTokens: 4_096 });
  });

  test("catalog JSON paths still work and retain architecture", async () => {
    const result = await fetchModelsFromSources(
      [{ url: "https://catalog.example/models", itemsPath: "models", idPath: "model.id", namePath: "display" }],
      undefined,
      async () => new Response(JSON.stringify({
        models: [{
          model: { id: "decider" },
          display: "Decider",
          architecture: { input_modalities: ["text"], output_modalities: ["decisions", "\u001b[31mterminal-injection", 17] },
        }],
      }), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    expect(result.models[0]).toMatchObject({
      id: "decider",
      name: "Decider",
      architecture: { input_modalities: ["text"], output_modalities: ["decisions"] },
    });
  });

  test("a failed modality listing does not discard successful listings", async () => {
    const result = await fetchModelsFromSources(
      buildModelSources("https://openrouter.ai/api/v1"),
      undefined,
      async (input) => {
        const url = new URL(String(input));
        if (url.searchParams.get("output_modalities") === "decisions") {
          return new Response("unavailable", { status: 503, statusText: "Service Unavailable" });
        }
        return new Response(JSON.stringify({ data: [{ id: "chat", name: "Chat" }] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      },
    );
    expect(result.models.map((model) => model.id)).toEqual(["chat"]);
    expect(result.warnings).toHaveLength(1);
    expect(result.warnings[0]).toContain("output_modalities=decisions");
  });

  test("decisions register as classifier-only with the System One handler", () => {
    const result = buildEntries([{
      id: "jev",
      architecture: { input_modalities: ["text"], output_modalities: ["decisions"] },
    }]);
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({ type: "classifier", api: "typesafe-system-one", input: ["text"] });
    const handlers = buildOperationHandlers(result.models);
    expect(typeof handlers.classifiers?.["typesafe-system-one"].classify).toBe("function");
    expect(handlers).not.toHaveProperty("images");

    const registry = createModels();
    registry.setProvider({
      id: "corral-local",
      name: "Corral Local",
      getModels: () => [],
      getAllModels: () => result.models,
    });
    expect(registry.getModelsOfType("classifier", "corral-local").map((model) => model.id)).toEqual(["jev"]);
  });

  test("mixed text and decisions outputs create separate operations with the same ID", () => {
    const result = buildEntries([{
      id: "dual",
      architecture: { input_modalities: ["text"], output_modalities: ["text", "decisions"] },
    }]);
    expect(result.models.map((model) => model.type)).toEqual(["chat", "classifier"]);
    expect(result.models.map((model) => model.id)).toEqual(["dual", "dual"]);
  });

  test("image output registers only with a compatible image API", () => {
    const model = {
      id: "image-model",
      architecture: { input_modalities: ["text", "image"], output_modalities: ["image", "text"] },
    };
    const withoutApi = buildEntries([model]);
    expect(withoutApi.models.map((entry) => entry.type)).toEqual(["chat"]);
    expect(withoutApi.skippedImageModels).toBe(1);

    const withApi = buildEntries([model], { imageApi: "openrouter-images" });
    expect(withApi.models.map((entry) => entry.type)).toEqual(["chat", "image"]);
    expect(withApi.models[1]).toMatchObject({
      api: "openrouter-images",
      input: ["text", "image"],
      output: ["image", "text"],
    });
    const handlers = buildOperationHandlers(withApi.models);
    expect(typeof handlers.images?.["openrouter-images"].generateImages).toBe("function");
  });

  test("OpenRouter image API is inferred only for its native base URL", () => {
    expect(resolveImageApi("https://openrouter.ai/api/v1")).toBe("openrouter-images");
    expect(resolveImageApi("http://corral.local/v1")).toBeUndefined();
    expect(resolveImageApi("http://corral.local/v1", "openrouter-images")).toBe("openrouter-images");
  });

  test("audio, embeddings, and rerank outputs never become chat models", () => {
    const result = buildEntries([{
      id: "non-chat",
      architecture: { input_modalities: ["text"], output_modalities: ["audio", "embeddings", "rerank"] },
    }]);
    expect(result.models).toEqual([]);
    expect(result.unsupportedModels).toBe(1);
    expect(result.unsupportedOutputModalities).toEqual(["audio", "embeddings", "rerank"]);
  });

  test("unsupported-only inputs do not become text chat or classifier models", () => {
    const result = buildEntries([{
      id: "audio-input",
      architecture: { input_modalities: ["audio"], output_modalities: ["text", "decisions"] },
    }]);
    expect(result.models).toEqual([]);
    expect(result.unsupportedModels).toBe(1);
    expect(result.unsupportedInputModalities).toEqual(["audio"]);
  });

  test("manual architecture overrides discovered output modality", () => {
    const result = buildEntries([{
      id: "configured",
      architecture: { input_modalities: ["text"], output_modalities: ["text"] },
    }], {
      overrides: {
        configured: {
          architecture: { output_modalities: ["decisions"] },
        },
      },
    });
    expect(result.models.map((model) => model.type)).toEqual(["classifier"]);
  });
});
