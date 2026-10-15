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
  imageApi?: "openrouter-images";
};

function buildEntries(
  models: Array<{
    id: string;
    name?: string;
    architecture?: { input_modalities?: string[]; output_modalities?: string[] };
  }>,
  options: { imageApi?: string; overrides?: Record<string, TestOverride> } = {},
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
    corralDetails: emptyCorralDetails(),
  });
}

describe("startup summary", () => {
  test("stays quiet when registration has no issues", () => {
    expect(formatStartupSummary({
      unavailableSources: 0,
      configuredFallbackProviders: 0,
      skippedProviders: 0,
      emptyProviders: 0,
      unsupportedModels: 0,
      skippedImageModels: 0,
    })).toEqual([]);
  });

  test("aggregates startup issues into one concise line", () => {
    expect(formatStartupSummary({
      unavailableSources: 3,
      configuredFallbackProviders: 1,
      skippedProviders: 1,
      emptyProviders: 0,
      unsupportedModels: 3,
      skippedImageModels: 1,
    })).toEqual([
      '[pi-dynamic-models] unavailable model sources: 3; configured fallback providers: 1; skipped providers: 1',
      '[pi-dynamic-models] unsupported models: 3; images requiring imageApi "openrouter-images": 1',
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
      ]));
      globalThis.fetch = (async (input, init) => {
        fetchCount++;
        const url = new URL(String(input));
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

      const ui = { setWidget: () => {} };
      const start = handlers.get("session_start");
      expect(start).toBeDefined();
      expect(start?.({ type: "session_start", reason: "startup" }, { ui })).toBeUndefined();
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

  test("merges architecture modalities across OpenRouter listings", async () => {
    const fetcher: typeof fetch = async (input) => {
      const url = new URL(String(input));
      const output = url.searchParams.get("output_modalities") ?? "text";
      return new Response(JSON.stringify({
        data: [{
          id: "jev",
          name: "Jev",
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
    expect(result.models).toEqual([{
      id: "jev",
      name: "Jev",
      architecture: {
        input_modalities: ["text"],
        output_modalities: ["text", "image", "decisions"],
      },
    }]);
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
