import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { NodeHttpHandler } from "@smithy/node-http-handler";
import { HttpsProxyAgent } from "https-proxy-agent";
import type { EmbeddingProvider, EmbeddingProviderOptions } from "./embeddings.js";

// ---------------------------------------------------------------------------
// Model registry — add new Bedrock embedding models here as they become GA
// ---------------------------------------------------------------------------

type BedrockModelConfig = {
  maxInputChars: number;
  defaultDimension: number;
  buildRequest: (text: string, dimension: number) => unknown;
  parseResponse: (body: unknown) => number[];
};

function buildNova2Request(text: string, dimension: number): unknown {
  return {
    schemaVersion: "nova-multimodal-embed-v1",
    taskType: "SINGLE_EMBEDDING",
    singleEmbeddingParams: {
      embeddingPurpose: "GENERIC_INDEX",
      embeddingDimension: dimension,
      text: {
        truncationMode: "END",
        value: text,
      },
    },
  };
}

function parseNova2Response(body: unknown): number[] {
  const payload = body as { embeddings?: Array<{ embedding?: number[] }> };
  return payload.embeddings?.[0]?.embedding ?? [];
}

const BEDROCK_MODEL_REGISTRY: Record<string, BedrockModelConfig> = {
  "amazon.nova-2-multimodal-embeddings-v1:0": {
    maxInputChars: 8192,
    defaultDimension: 1024,
    buildRequest: buildNova2Request,
    parseResponse: parseNova2Response,
  },
};

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

export const DEFAULT_BEDROCK_EMBEDDING_MODEL = "amazon.nova-2-multimodal-embeddings-v1:0";
export const DEFAULT_BEDROCK_REGION = "us-east-1";

export type BedrockEmbeddingClient = {
  modelId: string;
  region: string;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function normalizeBedrockModel(model: string): string {
  const trimmed = model.trim();
  if (!trimmed) {
    return DEFAULT_BEDROCK_EMBEDDING_MODEL;
  }
  if (trimmed.startsWith("bedrock/")) {
    return trimmed.slice("bedrock/".length);
  }
  return trimmed;
}

function resolveBedrockRegion(options: EmbeddingProviderOptions): string {
  const providerCfg = options.config.models?.providers?.["amazon-bedrock"] as
    | { region?: string }
    | undefined;
  return (
    providerCfg?.region?.trim() ||
    process.env.AWS_REGION?.trim() ||
    process.env.AWS_DEFAULT_REGION?.trim() ||
    DEFAULT_BEDROCK_REGION
  );
}

/**
 * Validate that AWS credentials are available (fail fast at startup).
 * The SDK uses the default credential chain at request time:
 *   1. Env vars: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN)
 *   2. SSO / profile credentials
 *   3. IMDS (EC2 instance role)
 *
 * We also accept AWS_BEARER_TOKEN_BEDROCK for backwards compat with the
 * upstream PR's Bearer token flow (used in local dev with SSO).
 */
function validateCredentials(): void {
  const hasIamCreds = !!(
    process.env.AWS_ACCESS_KEY_ID?.trim() && process.env.AWS_SECRET_ACCESS_KEY?.trim()
  );
  const hasBearer = !!process.env.AWS_BEARER_TOKEN_BEDROCK?.trim();
  const hasProfile = !!process.env.AWS_PROFILE?.trim();

  if (!hasIamCreds && !hasBearer && !hasProfile) {
    throw new Error(
      [
        'No API key found for provider "bedrock".',
        "Set AWS credentials via environment variables (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY),",
        "or AWS_PROFILE, or AWS_BEARER_TOKEN_BEDROCK for SSO.",
      ].join("\n"),
    );
  }
}

/**
 * Create a BedrockRuntimeClient with proxy support for Nitro Enclave.
 * Same pattern as bedrock-discovery.ts createProxyAwareBedrockClient and
 * pi-ai's amazon-bedrock.js runtime (the proven working path).
 */
function createBedrockRuntimeClient(region: string): BedrockRuntimeClient {
  const proxyUrl =
    process.env.HTTPS_PROXY ||
    process.env.https_proxy ||
    process.env.HTTP_PROXY ||
    process.env.http_proxy;
  if (proxyUrl) {
    const agent = new HttpsProxyAgent(proxyUrl);
    return new BedrockRuntimeClient({
      region,
      requestHandler: new NodeHttpHandler({
        httpAgent: agent,
        httpsAgent: agent,
      }),
    });
  }
  return new BedrockRuntimeClient({ region });
}

// ---------------------------------------------------------------------------
// Provider factory
// ---------------------------------------------------------------------------

export async function createBedrockEmbeddingProvider(
  options: EmbeddingProviderOptions,
): Promise<{ provider: EmbeddingProvider; client: BedrockEmbeddingClient }> {
  const modelId = normalizeBedrockModel(options.model);
  const modelConfig = BEDROCK_MODEL_REGISTRY[modelId];

  if (!modelConfig) {
    const supported = Object.keys(BEDROCK_MODEL_REGISTRY).join(", ");
    throw new Error(
      `Unsupported Bedrock embedding model: "${modelId}". Supported models: ${supported}`,
    );
  }

  // Fail fast if no credentials are available
  validateCredentials();

  const region = resolveBedrockRegion(options);
  const runtimeClient = createBedrockRuntimeClient(region);
  const client: BedrockEmbeddingClient = { modelId, region };

  const { defaultDimension, buildRequest, parseResponse } = modelConfig;

  const embedSingle = async (text: string): Promise<number[]> => {
    const body = buildRequest(text, defaultDimension);
    const command = new InvokeModelCommand({
      modelId,
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify(body),
    });
    const response = await runtimeClient.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));
    return parseResponse(responseBody);
  };

  return {
    provider: {
      id: "bedrock",
      model: modelId,
      embedQuery: embedSingle,
      embedBatch: (texts) => Promise.all(texts.map(embedSingle)),
    },
    client,
  };
}
