import { createModels } from "@earendil-works/pi-ai";
import { describe, expect, test } from "bun:test";
import {
  buildModelEntries,
  buildModelSources,
  buildOperationHandlers,
  fetchModelsFromSources,
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
