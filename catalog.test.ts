import { describe, expect, test } from "bun:test";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { AnyModel } from "@earendil-works/pi-ai";
import { buildModelEntries, buildOperationHandlers, hydrateOpenRouter, resolveImageApi } from "./index.ts";
import { generateImages } from "./image-api.ts";

const image = {
  type: "image" as const, provider: "local-llm", id: "arbitrary-id", name: "Image", api: "openai-images",
  baseUrl: "http://corral.test/v1", input: ["text" as const], output: ["image" as const],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

describe("corral image operation", () => {
  test("declared image output routes through the configured adapter, never model-name guessing", () => {
    const architecture = { input_modalities: ["text"], output_modalities: ["image"] };
    const registration = buildModelEntries({
      ids: ["arbitrary-id"], provider: "local-llm", baseUrl: image.baseUrl, api: "openai-completions",
      imageApi: resolveImageApi(image.baseUrl, "openai-images"),
      discoveredById: new Map([["arbitrary-id", { id: "arbitrary-id", architecture }]]), corralDetails: new Map(),
    });
    expect(registration.models.map((m) => m.type)).toEqual(["image"]);
    expect(registration.models[0].api).toBe("openai-images");
    expect(typeof buildOperationHandlers(registration.models).images?.["openai-images"].generateImages).toBe("function");
  });

  test("corral detail architecture is used when the first catalog lacks it", () => {
    const registration = buildModelEntries({
      ids: ["decider"], provider: "local-llm", baseUrl: image.baseUrl, api: "openai-completions",
      discoveredById: new Map([["decider", { id: "decider" }]]),
      corralDetails: new Map([["decider", { id: "decider", context_size: 32000,
        architecture: { input_modalities: ["text"], output_modalities: ["decisions"] } }]]),
    });
    expect(registration.models.map((m) => m.type)).toEqual(["classifier"]);
    expect(registration.models[0]).toMatchObject({ api: "typesafe-system-one", contextWindow: 32000 });
  });

  test("OpenAI image adapter preserves hooks, auth, prompt, cancellation and malformed-output failures", async () => {
    let request: any;
    let status: number | undefined;
    const result = await generateImages(image, { input: [{ type: "text", text: "draw M" }] }, {
      apiKey: "test-key", headers: { "X-Test": "yes" },
      onPayload: (payload) => ({ ...(payload as object), size: "1024x1024" }),
      onResponse: (response) => { status = response.status; },
      fetch: (async (url: any, init: any) => {
        request = { url, ...init, body: JSON.parse(init.body) };
        return Response.json({ data: [{ b64_json: "aW1hZ2U=" }] });
      }) as typeof fetch,
    });
    expect(result.stopReason).toBe("stop");
    expect(request.url).toBe("http://corral.test/v1/images/generations");
    expect(request.headers.get("Authorization")).toBe("Bearer test-key");
    expect(request.headers.get("X-Test")).toBe("yes");
    expect(request.body).toEqual({ model: "arbitrary-id", prompt: "draw M", n: 1, response_format: "b64_json", size: "1024x1024" });
    expect(result.output).toEqual([{ type: "image", data: "aW1hZ2U=", mimeType: "image/png" }]);
    expect(status).toBe(200);
    const empty = await generateImages(image, { input: [] }, { fetch: (async () => Response.json({ data: [] })) as any });
    expect(empty.stopReason).toBe("error");
    const controller = new AbortController(); controller.abort();
    const aborted = await generateImages(image, { input: [] }, {
      signal: controller.signal, fetch: (async (_: any, init: any) => { init.signal.throwIfAborted(); }) as any,
    });
    expect(aborted.stopReason).toBe("aborted");
  });
});

describe("native OpenRouter dynamic catalogs", () => {
  test("operation discovery preserves native auth, chat and subsequent catalog refreshes in Pi's runtime", async () => {
    const oldFetch = globalThis.fetch;
    const dir = mkdtempSync(join(tmpdir(), "dynamic-runtime-"));
    try {
      const runtime = await ModelRuntime.create({
        authPath: join(dir, "auth.json"), modelsPath: null,
        modelsStorePath: join(dir, "models-store.json"), refreshOnCreate: false,
      });
      const original = runtime.getProvider("openrouter")!;
      let chats = [{
        type: "chat" as const, provider: "openrouter", id: "chat", name: "Chat",
        baseUrl: "https://openrouter.ai/api/v1", api: "openai-completions",
        reasoning: false, input: ["text" as const], contextWindow: 999, maxTokens: 64,
        cost: { input: 10, output: 8, cacheRead: 0, cacheWrite: 0 },
      }];
      const old = { ...image, provider: "openrouter", id: "new/image", cost: { input: 4, output: 9, cacheRead: 0, cacheWrite: 0 } };
      const native = {
        ...original,
        getModels: () => chats,
        getAllModels: () => [...chats, old],
        refreshModels: async (context: any) => {
          if (!context.allowNetwork) return;
          await context.publish({ update: () => {
            chats = [{ ...chats[0], contextWindow: 5555, cost: { ...chats[0].cost, input: 12345 } }];
          } });
        },
      };
      runtime.registerNativeProvider(native);
      let registered: any;
      const urls: string[] = [];
      globalThis.fetch = (async (url: any) => {
        urls.push(String(url));
        const modality = new URL(url).searchParams.get("output_modalities")!;
        return Response.json({ data: [{ id: `new/${modality}`, architecture: { input_modalities: ["text"], output_modalities: [modality] } }] });
      }) as any;
      await hydrateOpenRouter({
        registerProvider: (provider: any) => { registered = provider; runtime.registerNativeProvider(provider); },
      } as any, { modelRegistry: { getProvider: (id: string) => runtime.getProvider(id) } } as any);

      expect(urls).toEqual([
        "https://openrouter.ai/api/v1/models?output_modalities=image",
        "https://openrouter.ai/api/v1/models?output_modalities=decisions",
      ]);
      expect(runtime.getAllModels("openrouter").map((m: AnyModel) => [m.type, m.id])).toEqual([
        ["chat", "chat"], ["image", "new/image"], ["classifier", "new/decisions"],
      ]);
      expect(runtime.getModel("openrouter", "chat")).toBe(chats[0]);
      expect(runtime.getModelOfType("image", "openrouter", "new/image")?.cost).toEqual(old.cost);
      expect(registered.auth).toBe(native.auth);
      expect(registered.stream).toBe(native.stream);
      expect(registered.generateImages).toBe(native.generateImages);
      await runtime.refresh({ providers: ["openrouter"], allowNetwork: true, force: true });
      expect(runtime.getModel("openrouter", "chat")?.cost.input).toBe(12345);
      expect(runtime.getModel("openrouter", "chat")?.contextWindow).toBe(5555);
    } finally {
      globalThis.fetch = oldFetch;
      rmSync(dir, { recursive: true });
    }
  });
});
