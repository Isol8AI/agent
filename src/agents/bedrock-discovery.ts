import {
  BedrockClient,
  ListFoundationModelsCommand,
  ListInferenceProfilesCommand,
  type ListFoundationModelsCommandOutput,
} from "@aws-sdk/client-bedrock";
import { NodeHttpHandler } from "@smithy/node-http-handler";
import { HttpsProxyAgent } from "https-proxy-agent";
import type { BedrockDiscoveryConfig, ModelDefinitionConfig } from "../config/types.js";
import { createSubsystemLogger } from "../logging/subsystem.js";

const log = createSubsystemLogger("bedrock-discovery");

/**
 * Create a BedrockClient that routes through an HTTP proxy when proxy env vars are set.
 *
 * Inside a Nitro Enclave there's no DNS or direct internet — all traffic must go
 * through the vsock HTTP CONNECT proxy. The default BedrockClient doesn't respect
 * HTTP_PROXY env vars, so we configure NodeHttpHandler with https-proxy-agent.
 */
function createProxyAwareBedrockClient(region: string): BedrockClient {
  const proxyUrl =
    process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
  if (proxyUrl) {
    const agent = new HttpsProxyAgent(proxyUrl);
    return new BedrockClient({
      region,
      requestHandler: new NodeHttpHandler({
        httpAgent: agent,
        httpsAgent: agent,
      }),
    });
  }
  return new BedrockClient({ region });
}

const DEFAULT_REFRESH_INTERVAL_SECONDS = 3600;
const DEFAULT_CONTEXT_WINDOW = 32000;
const DEFAULT_MAX_TOKENS = 4096;
const DEFAULT_COST = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
};

type BedrockModelSummary = NonNullable<ListFoundationModelsCommandOutput["modelSummaries"]>[number];

type BedrockDiscoveryCacheEntry = {
  expiresAt: number;
  value?: ModelDefinitionConfig[];
  inFlight?: Promise<ModelDefinitionConfig[]>;
};

const discoveryCache = new Map<string, BedrockDiscoveryCacheEntry>();
let hasLoggedBedrockError = false;

function normalizeProviderFilter(filter?: string[]): string[] {
  if (!filter || filter.length === 0) {
    return [];
  }
  const normalized = new Set(
    filter.map((entry) => entry.trim().toLowerCase()).filter((entry) => entry.length > 0),
  );
  return Array.from(normalized).toSorted();
}

function buildCacheKey(params: {
  region: string;
  providerFilter: string[];
  refreshIntervalSeconds: number;
  defaultContextWindow: number;
  defaultMaxTokens: number;
}): string {
  return JSON.stringify(params);
}

function includesTextModalities(modalities?: Array<string>): boolean {
  return (modalities ?? []).some((entry) => entry.toLowerCase() === "text");
}

function isActive(summary: BedrockModelSummary): boolean {
  const status = summary.modelLifecycle?.status;
  return typeof status === "string" ? status.toUpperCase() === "ACTIVE" : false;
}

function mapInputModalities(summary: BedrockModelSummary): Array<"text" | "image"> {
  const inputs = summary.inputModalities ?? [];
  const mapped = new Set<"text" | "image">();
  for (const modality of inputs) {
    const lower = modality.toLowerCase();
    if (lower === "text") {
      mapped.add("text");
    }
    if (lower === "image") {
      mapped.add("image");
    }
  }
  if (mapped.size === 0) {
    mapped.add("text");
  }
  return Array.from(mapped);
}

function inferReasoningSupport(summary: BedrockModelSummary): boolean {
  const haystack = `${summary.modelId ?? ""} ${summary.modelName ?? ""}`.toLowerCase();
  return haystack.includes("reasoning") || haystack.includes("thinking");
}

function resolveDefaultContextWindow(config?: BedrockDiscoveryConfig): number {
  const value = Math.floor(config?.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW);
  return value > 0 ? value : DEFAULT_CONTEXT_WINDOW;
}

function resolveDefaultMaxTokens(config?: BedrockDiscoveryConfig): number {
  const value = Math.floor(config?.defaultMaxTokens ?? DEFAULT_MAX_TOKENS);
  return value > 0 ? value : DEFAULT_MAX_TOKENS;
}

function matchesProviderFilter(summary: BedrockModelSummary, filter: string[]): boolean {
  if (filter.length === 0) {
    return true;
  }
  const providerName =
    summary.providerName ??
    (typeof summary.modelId === "string" ? summary.modelId.split(".")[0] : undefined);
  const normalized = providerName?.trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return filter.includes(normalized);
}

function shouldIncludeSummary(summary: BedrockModelSummary, filter: string[]): boolean {
  if (!summary.modelId?.trim()) {
    return false;
  }
  if (!matchesProviderFilter(summary, filter)) {
    return false;
  }
  if (summary.responseStreamingSupported !== true) {
    return false;
  }
  if (!includesTextModalities(summary.outputModalities)) {
    return false;
  }
  if (!isActive(summary)) {
    return false;
  }
  return true;
}

function toModelDefinition(
  summary: BedrockModelSummary,
  defaults: { contextWindow: number; maxTokens: number },
  inferenceProfileId?: string,
): ModelDefinitionConfig {
  const baseId = summary.modelId?.trim() ?? "";
  // Use inference profile ID for invocation if available
  const id = inferenceProfileId ?? baseId;
  return {
    id,
    name: summary.modelName?.trim() || id,
    reasoning: inferReasoningSupport(summary),
    input: mapInputModalities(summary),
    cost: DEFAULT_COST,
    contextWindow: defaults.contextWindow,
    maxTokens: defaults.maxTokens,
  };
}

/**
 * Build a mapping from base model ID to inference profile ID.
 * Some Bedrock models require inference profile IDs (e.g. us.anthropic.claude-...)
 * for invocation and don't support on-demand throughput with base model IDs.
 */
async function buildInferenceProfileMap(client: BedrockClient): Promise<Map<string, string>> {
  const mapping = new Map<string, string>();
  try {
    let nextToken: string | undefined;
    do {
      const response = await client.send(new ListInferenceProfilesCommand({ nextToken }));
      for (const profile of response.inferenceProfileSummaries ?? []) {
        const profileId = profile.inferenceProfileId?.trim();
        if (!profileId) continue;
        for (const modelRef of profile.models ?? []) {
          const arn = modelRef.modelArn ?? "";
          // ARN: arn:aws:bedrock:region::foundation-model/base-model-id
          const idx = arn.indexOf("foundation-model/");
          if (idx >= 0) {
            const baseId = arn.slice(idx + "foundation-model/".length).trim();
            if (baseId) {
              mapping.set(baseId, profileId);
            }
          }
        }
      }
      nextToken = response.nextToken;
    } while (nextToken);
  } catch {
    // Non-fatal: if ListInferenceProfiles fails, we'll use base IDs
  }
  return mapping;
}

export function resetBedrockDiscoveryCacheForTest(): void {
  discoveryCache.clear();
  hasLoggedBedrockError = false;
}

export async function discoverBedrockModels(params: {
  region: string;
  config?: BedrockDiscoveryConfig;
  now?: () => number;
  clientFactory?: (region: string) => BedrockClient;
}): Promise<ModelDefinitionConfig[]> {
  const refreshIntervalSeconds = Math.max(
    0,
    Math.floor(params.config?.refreshInterval ?? DEFAULT_REFRESH_INTERVAL_SECONDS),
  );
  const providerFilter = normalizeProviderFilter(params.config?.providerFilter);
  const defaultContextWindow = resolveDefaultContextWindow(params.config);
  const defaultMaxTokens = resolveDefaultMaxTokens(params.config);
  const cacheKey = buildCacheKey({
    region: params.region,
    providerFilter,
    refreshIntervalSeconds,
    defaultContextWindow,
    defaultMaxTokens,
  });
  const now = params.now?.() ?? Date.now();

  if (refreshIntervalSeconds > 0) {
    const cached = discoveryCache.get(cacheKey);
    if (cached?.value && cached.expiresAt > now) {
      return cached.value;
    }
    if (cached?.inFlight) {
      return cached.inFlight;
    }
  }

  const clientFactory = params.clientFactory ?? createProxyAwareBedrockClient;
  const client = clientFactory(params.region);

  const discoveryPromise = (async () => {
    const [response, profileMap] = await Promise.all([
      client.send(new ListFoundationModelsCommand({})),
      buildInferenceProfileMap(client),
    ]);
    const discovered: ModelDefinitionConfig[] = [];
    const registeredIds = new Set<string>();
    for (const summary of response.modelSummaries ?? []) {
      if (!shouldIncludeSummary(summary, providerFilter)) {
        continue;
      }
      const baseId = summary.modelId?.trim() ?? "";
      const profileId = profileMap.get(baseId);
      const def = toModelDefinition(
        summary,
        { contextWindow: defaultContextWindow, maxTokens: defaultMaxTokens },
        profileId,
      );
      discovered.push(def);
      registeredIds.add(def.id);
      // Also register the base ID so ModelRegistry.find() works with either
      if (profileId && !registeredIds.has(baseId)) {
        discovered.push(
          toModelDefinition(summary, {
            contextWindow: defaultContextWindow,
            maxTokens: defaultMaxTokens,
          }),
        );
        registeredIds.add(baseId);
      }
    }
    return discovered.toSorted((a, b) => a.name.localeCompare(b.name));
  })();

  if (refreshIntervalSeconds > 0) {
    discoveryCache.set(cacheKey, {
      expiresAt: now + refreshIntervalSeconds * 1000,
      inFlight: discoveryPromise,
    });
  }

  try {
    const value = await discoveryPromise;
    if (refreshIntervalSeconds > 0) {
      discoveryCache.set(cacheKey, {
        expiresAt: now + refreshIntervalSeconds * 1000,
        value,
      });
    }
    return value;
  } catch (error) {
    if (refreshIntervalSeconds > 0) {
      discoveryCache.delete(cacheKey);
    }
    if (!hasLoggedBedrockError) {
      hasLoggedBedrockError = true;
      log.warn(`Failed to list models: ${String(error)}`);
    }
    return [];
  }
}
