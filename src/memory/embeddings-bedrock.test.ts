import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createBedrockEmbeddingProvider,
  DEFAULT_BEDROCK_EMBEDDING_MODEL,
  normalizeBedrockModel,
} from "./embeddings-bedrock.js";

// vi.hoisted runs before vi.mock hoisting, so these are available in the factory
const { mockSend, MockBedrockRuntimeClient, MockInvokeModelCommand } = vi.hoisted(() => {
  const mockSend = vi.fn();
  // Must be a class or function (not arrow) to support `new`
  class MockBedrockRuntimeClient {
    send = mockSend;
  }
  class MockInvokeModelCommand {
    input: unknown;
    constructor(input: unknown) {
      this.input = input;
    }
  }
  return { mockSend, MockBedrockRuntimeClient, MockInvokeModelCommand };
});

vi.mock("@aws-sdk/client-bedrock-runtime", () => ({
  BedrockRuntimeClient: MockBedrockRuntimeClient,
  InvokeModelCommand: MockInvokeModelCommand,
}));

// Mock proxy deps (not needed in tests)
vi.mock("@smithy/node-http-handler", () => ({ NodeHttpHandler: vi.fn() }));
vi.mock("https-proxy-agent", () => ({ HttpsProxyAgent: vi.fn() }));

function mockBedrockResponse(embedding = [0.1, 0.2, 0.3]) {
  mockSend.mockResolvedValue({
    body: new TextEncoder().encode(
      JSON.stringify({ embeddings: [{ embeddingType: "TEXT", embedding }] }),
    ),
  });
}

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllEnvs();
});

describe("normalizeBedrockModel", () => {
  it("returns default for empty string", () => {
    expect(normalizeBedrockModel("")).toBe(DEFAULT_BEDROCK_EMBEDDING_MODEL);
    expect(normalizeBedrockModel("  ")).toBe(DEFAULT_BEDROCK_EMBEDDING_MODEL);
  });

  it("strips bedrock/ prefix", () => {
    expect(normalizeBedrockModel("bedrock/amazon.nova-2-multimodal-embeddings-v1:0")).toBe(
      "amazon.nova-2-multimodal-embeddings-v1:0",
    );
  });

  it("passes through bare model id", () => {
    expect(normalizeBedrockModel("amazon.nova-2-multimodal-embeddings-v1:0")).toBe(
      "amazon.nova-2-multimodal-embeddings-v1:0",
    );
  });
});

describe("createBedrockEmbeddingProvider", () => {
  it("uses default region when AWS_REGION not set", async () => {
    vi.stubEnv("AWS_ACCESS_KEY_ID", "test-key");
    vi.stubEnv("AWS_SECRET_ACCESS_KEY", "test-secret");
    mockBedrockResponse();

    const { client } = await createBedrockEmbeddingProvider({
      config: {} as never,
      provider: "bedrock",
      model: "",
      fallback: "none",
    });

    expect(client.region).toBe("us-east-1");
  });

  it("uses AWS_REGION env var for region", async () => {
    vi.stubEnv("AWS_ACCESS_KEY_ID", "test-key");
    vi.stubEnv("AWS_SECRET_ACCESS_KEY", "test-secret");
    vi.stubEnv("AWS_REGION", "ap-southeast-1");
    mockBedrockResponse();

    const { client } = await createBedrockEmbeddingProvider({
      config: {} as never,
      provider: "bedrock",
      model: "",
      fallback: "none",
    });

    expect(client.region).toBe("ap-southeast-1");
  });

  it("sends correct Nova 2 request body via InvokeModelCommand", async () => {
    vi.stubEnv("AWS_ACCESS_KEY_ID", "test-key");
    vi.stubEnv("AWS_SECRET_ACCESS_KEY", "test-secret");
    mockBedrockResponse();

    const { provider } = await createBedrockEmbeddingProvider({
      config: {} as never,
      provider: "bedrock",
      model: "",
      fallback: "none",
    });

    await provider.embedQuery("test input");

    // Verify the command passed to client.send()
    const command = mockSend.mock.calls[0][0] as InstanceType<typeof MockInvokeModelCommand>;
    const input = command.input as { modelId: string; body: string };
    expect(input.modelId).toBe(DEFAULT_BEDROCK_EMBEDDING_MODEL);
    const body = JSON.parse(input.body);
    expect(body).toMatchObject({
      schemaVersion: "nova-multimodal-embed-v1",
      taskType: "SINGLE_EMBEDDING",
      singleEmbeddingParams: {
        embeddingPurpose: "GENERIC_INDEX",
        embeddingDimension: 1024,
        text: { truncationMode: "END", value: "test input" },
      },
    });
  });

  it("parses Nova 2 response correctly", async () => {
    vi.stubEnv("AWS_ACCESS_KEY_ID", "test-key");
    vi.stubEnv("AWS_SECRET_ACCESS_KEY", "test-secret");
    mockBedrockResponse([0.1, 0.2, 0.3]);

    const { provider } = await createBedrockEmbeddingProvider({
      config: {} as never,
      provider: "bedrock",
      model: "",
      fallback: "none",
    });

    const result = await provider.embedQuery("hello");
    expect(result).toEqual([0.1, 0.2, 0.3]);
  });

  it("embedBatch calls invoke once per text", async () => {
    vi.stubEnv("AWS_ACCESS_KEY_ID", "test-key");
    vi.stubEnv("AWS_SECRET_ACCESS_KEY", "test-secret");
    mockBedrockResponse([0.5, 0.6]);

    const { provider } = await createBedrockEmbeddingProvider({
      config: {} as never,
      provider: "bedrock",
      model: "",
      fallback: "none",
    });

    const results = await provider.embedBatch(["a", "b", "c"]);
    expect(mockSend).toHaveBeenCalledTimes(3);
    expect(results).toHaveLength(3);
  });

  it("throws when no AWS credentials are available", async () => {
    delete process.env.AWS_ACCESS_KEY_ID;
    delete process.env.AWS_SECRET_ACCESS_KEY;
    delete process.env.AWS_BEARER_TOKEN_BEDROCK;
    delete process.env.AWS_PROFILE;

    await expect(
      createBedrockEmbeddingProvider({
        config: {} as never,
        provider: "bedrock",
        model: "",
        fallback: "none",
      }),
    ).rejects.toThrow('No API key found for provider "bedrock"');
  });

  it("accepts AWS_BEARER_TOKEN_BEDROCK as valid credentials", async () => {
    delete process.env.AWS_ACCESS_KEY_ID;
    delete process.env.AWS_SECRET_ACCESS_KEY;
    vi.stubEnv("AWS_BEARER_TOKEN_BEDROCK", "sso-token");
    mockBedrockResponse();

    const { provider } = await createBedrockEmbeddingProvider({
      config: {} as never,
      provider: "bedrock",
      model: "",
      fallback: "none",
    });

    expect(provider.id).toBe("bedrock");
  });

  it("throws for unsupported model", async () => {
    vi.stubEnv("AWS_ACCESS_KEY_ID", "test-key");
    vi.stubEnv("AWS_SECRET_ACCESS_KEY", "test-secret");

    await expect(
      createBedrockEmbeddingProvider({
        config: {} as never,
        provider: "bedrock",
        model: "amazon.titan-embed-text-v2:0",
        fallback: "none",
      }),
    ).rejects.toThrow("Unsupported Bedrock embedding model");
  });

  it("exposes provider id and model", async () => {
    vi.stubEnv("AWS_ACCESS_KEY_ID", "test-key");
    vi.stubEnv("AWS_SECRET_ACCESS_KEY", "test-secret");
    mockBedrockResponse();

    const { provider, client } = await createBedrockEmbeddingProvider({
      config: {} as never,
      provider: "bedrock",
      model: "",
      fallback: "none",
    });

    expect(provider.id).toBe("bedrock");
    expect(provider.model).toBe(DEFAULT_BEDROCK_EMBEDDING_MODEL);
    expect(client.modelId).toBe(DEFAULT_BEDROCK_EMBEDDING_MODEL);
  });
});
