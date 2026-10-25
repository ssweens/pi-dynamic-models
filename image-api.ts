import type { AssistantImages, ImagesFunction, ProviderImages } from "@earendil-works/pi-ai";

/** OpenAI-compatible image generation for servers exposing /v1/images/generations. */
export const generateImages: ImagesFunction = async (model, context, options) => {
  const result: AssistantImages = {
    api: model.api, provider: model.provider, model: model.id,
    output: [], stopReason: "stop", timestamp: Date.now(),
  };
  try {
    if (context.input.some((block) => block.type === "image")) {
      throw new Error("openai-images supports text-to-image generation only.");
    }
    const prompt = context.input.filter((block) => block.type === "text").map((block) => block.text).join("\n");
    const payload = { model: model.id, prompt, n: 1, response_format: "b64_json" };
    const body = await options?.onPayload?.(payload, model) ?? payload;
    const headers = new Headers({ "Content-Type": "application/json", ...model.headers });
    if (options?.apiKey && options.apiKey !== "none") headers.set("Authorization", `Bearer ${options.apiKey}`);
    for (const [key, value] of Object.entries(options?.headers ?? {})) {
      if (value === null) headers.delete(key);
      else headers.set(key, value);
    }
    const signal = options?.timeoutMs === undefined ? options?.signal : options.signal
      ? AbortSignal.any([options.signal, AbortSignal.timeout(options.timeoutMs)])
      : AbortSignal.timeout(options.timeoutMs);
    const response = await (options?.fetch ?? fetch)(`${model.baseUrl.replace(/\/+$/u, "")}/images/generations`, {
      method: "POST", headers, body: JSON.stringify(body), signal,
    });
    await options?.onResponse?.({ status: response.status, headers: Object.fromEntries(response.headers) }, model);
    if (!response.ok) throw new Error(`Image generation failed (${response.status}): ${await response.text()}`);
    const data = await response.json() as { data?: Array<{ b64_json?: string }> };
    for (const image of data.data ?? []) {
      if (typeof image.b64_json === "string" && image.b64_json) {
        result.output.push({ type: "image", data: image.b64_json, mimeType: "image/png" });
      }
    }
    if (!result.output.length) throw new Error("Image generation returned no base64 image data.");
  } catch (error) {
    result.stopReason = options?.signal?.aborted ? "aborted" : "error";
    result.errorMessage = error instanceof Error ? error.message : String(error);
  }
  return result;
};

export const openAIImagesApi = (): ProviderImages => ({ generateImages });
