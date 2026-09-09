import { beforeEach, describe, expect, it, vi } from "vitest";
import { createDeferred } from "../../../../test/helpers/promise.js";
import {
  createPluginMetadataSnapshot,
  makeRegistry,
} from "../../../config/plugin-auto-enable.test-helpers.js";
import { createDiagnosticTraceContext } from "../../../infra/diagnostic-trace-context.js";
import { setPluginToolMeta } from "../../../plugins/tool-metadata.js";
import { materializeBundleMcpToolsForRun } from "../../agent-bundle-mcp-materialize.js";
import type { McpToolCatalog, SessionMcpRuntime } from "../../agent-bundle-mcp-types.js";
import { resolveConversationCapabilityProfile } from "../../conversation-capability-profile.js";
import { withPrivateRoomExecution } from "../../private-room-execution.js";
import { createAgentCleanupScope } from "../../run-cleanup-timeout.js";
import { createStubTool } from "../../test-helpers/agent-tool-stubs.js";
import { attachToolAllowlistIntersection } from "../../tool-policy.js";
import {
  createToolSearchCatalogRef,
  createToolSearchTools,
  resolveToolSearchConfig,
  clearToolSearchCatalog,
} from "../../tool-search.js";

const mocks = vi.hoisted(() => ({
  createBundleLspToolRuntime: vi.fn(),
  acquireSessionMcpRuntime: vi.fn(),
  materializeBundleMcpToolsForRun: vi.fn(),
  applyFinalEffectiveToolPolicy: vi.fn(),
  filterRuntimeCompatibleTools: vi.fn(),
}));

vi.mock("../../agent-bundle-lsp-runtime.js", () => ({
  createBundleLspToolRuntime: mocks.createBundleLspToolRuntime,
}));

vi.mock("../../agent-bundle-mcp-tools.js", () => ({
  acquireSessionMcpRuntime: mocks.acquireSessionMcpRuntime,
  materializeBundleMcpToolsForRun: mocks.materializeBundleMcpToolsForRun,
}));

vi.mock("../../runtime-plan/tools.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../runtime-plan/tools.js")>()),
  normalizeAgentRuntimeTools: vi.fn(({ tools }: { tools: unknown[] }) => [...tools]),
}));

vi.mock("../../local-model-lean.js", () => ({
  filterLocalModelLeanTools: vi.fn(({ tools }: { tools: unknown[] }) => tools),
}));

vi.mock("../../tool-schema-projection.js", () => ({
  filterRuntimeCompatibleTools: mocks.filterRuntimeCompatibleTools,
}));

vi.mock("../effective-tool-policy.js", () => ({
  applyFinalEffectiveToolPolicy: mocks.applyFinalEffectiveToolPolicy,
}));

import { prepareEmbeddedAttemptBundleTools } from "./attempt-bundle-tools.js";
import { createPromptBuildToolPolicy } from "./attempt-prompt-support.js";
import { createAttemptSetupFixture } from "./attempt-setup.test-support.js";
import { prepareEmbeddedAttemptToolCatalog } from "./attempt-tool-catalog.js";

describe("prepareEmbeddedAttemptBundleTools", () => {
  it("does not acquire external MCP or LSP runtimes for a private execution", async () => {
    const input = createInput([], []);
    const result = await withPrivateRoomExecution(
      {
        agentId: "main",
        rootExecutionId: "root",
        runId: "run",
        hopCount: 0,
        sessionKey: "agent:main:room",
        sessionId: "room",
        inputMessageId: "input",
        assertCurrent: () => {},
        close: () => {},
      },
      () =>
        prepareEmbeddedAttemptBundleTools(
          input as Parameters<typeof prepareEmbeddedAttemptBundleTools>[0],
        ),
    );
    expect(result.bundleMcpRuntime).toBeUndefined();
    expect(result.bundleLspRuntime).toBeUndefined();
    expect(mocks.acquireSessionMcpRuntime).not.toHaveBeenCalled();
    expect(mocks.createBundleLspToolRuntime).not.toHaveBeenCalled();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mocks.createBundleLspToolRuntime.mockReset().mockResolvedValue(undefined);
    mocks.acquireSessionMcpRuntime.mockReset().mockResolvedValue(undefined);
    mocks.materializeBundleMcpToolsForRun.mockReset().mockResolvedValue(undefined);
    mocks.applyFinalEffectiveToolPolicy
      .mockReset()
      .mockImplementation(({ bundledTools }: { bundledTools: unknown[] }) => bundledTools);
    mocks.filterRuntimeCompatibleTools
      .mockReset()
      .mockImplementation((tools: unknown[]) => ({ tools, diagnostics: [] }));
  });

  function createInput(inheritedToolAllowlist: string[], toolsRaw: unknown[]) {
    return {
      agentDir: "/tmp/agent",
      attempt: {
        config: {},
        model: {},
        modelId: "model",
        provider: "provider",
        runId: "run",
        runtimePlan: {},
        sessionId: "session",
      },
      setup: createAttemptSetupFixture(),
      isRawModelRun: false,
      preparedToolBase: {
        cronCreatorToolAllowlist: [],
        effectiveToolsAllow: undefined,
        inheritedToolAllowlist,
        localModelLeanPreserveToolNames: [],
        runtimeCapabilityProfile: undefined,
        toolsEnabled: true,
        toolsRaw,
      },
    } as unknown as Parameters<typeof prepareEmbeddedAttemptBundleTools>[0];
  }

  it.each([false, true])(
    "blocks cold non-directory preparation (Code Mode=%s)",
    async (codeMode) => {
      const input = createInput([], []);
      input.attempt.config = {
        plugins: { enabled: false },
        mcp: { servers: { probe: { command: "fixture" } } },
      };
      input.attempt.runtimePlan = undefined;
      Object.assign(input.preparedToolBase, {
        codeModeControlsEnabledForRun: codeMode,
        toolSearchControlsEnabledForRun: false,
      });
      const deferred = createDeferred<McpToolCatalog>();
      const entered = createDeferred();
      const callTool = vi.fn(async () => ({
        content: [{ type: "text" as const, text: "allowed-cold" }],
      }));
      const runtime: SessionMcpRuntime = {
        sessionId: "session",
        workspaceDir: "/tmp/workspace",
        configFingerprint: "cold",
        createdAt: 0,
        lastUsedAt: 0,
        markUsed() {},
        peekCatalog: () => null,
        getCatalog: () => {
          entered.resolve();
          return deferred.promise;
        },
        callTool,
        dispose: async () => {},
        joinCleanup: async () => {},
        acquireLease: () => () => {},
      };
      mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime });
      mocks.materializeBundleMcpToolsForRun.mockImplementation(materializeBundleMcpToolsForRun);
      let prepared = false;
      const pending = prepareEmbeddedAttemptBundleTools(input).then((result) => {
        prepared = true;
        return result;
      });
      await entered.promise;
      expect(prepared).toBe(false);
      expect(mocks.materializeBundleMcpToolsForRun).toHaveBeenCalledWith(
        expect.objectContaining({ nonBlocking: false }),
      );
      deferred.resolve({
        version: 1,
        generatedAt: 1,
        servers: { probe: { serverName: "probe", launchSummary: "probe", toolCount: 2 } },
        tools: ["allowed", "denied"].map((toolName) => ({
          serverName: "probe",
          safeServerName: "probe",
          toolName,
          description: toolName,
          fallbackDescription: toolName,
          inputSchema: { type: "object", properties: {} },
        })),
      });
      const bundle = await pending;
      try {
        let activeToolNames = bundle.uncompactedEffectiveTools.map((tool) => tool.name);
        const policy = createPromptBuildToolPolicy({
          session: {
            getActiveToolNames: () => activeToolNames,
            setActiveToolsByName: (names) => {
              activeToolNames = names;
            },
          },
          effectiveTools: bundle.uncompactedEffectiveTools,
          uncompactedEffectiveTools: bundle.uncompactedEffectiveTools,
          tools: bundle.tools,
          codeModeControlsEnabled: false,
        });
        const surface = policy.apply(["probe__allowed"]);
        expect(surface.effectiveTools.map((tool) => tool.name)).toEqual(["probe__allowed"]);
        expect(activeToolNames).toEqual(["probe__allowed"]);
        const allowed = surface.effectiveTools.find((tool) => tool.name === "probe__allowed");
        if (!allowed) {
          throw new Error("Expected allowed cold MCP tool");
        }
        expect(JSON.stringify(await allowed.execute("allowed", {}))).toContain("allowed-cold");
        expect(callTool).toHaveBeenCalledExactlyOnceWith("probe", "allowed", {});
      } finally {
        await bundle.bundleMcpRuntime?.dispose();
      }
    },
  );

  it.each([false, true])(
    "retains prompt policy in the original cold directory (allow MCP=%s)",
    async (allowMcp) => {
      const config = {
        plugins: { enabled: false },
        mcp: { servers: { probe: { command: "fixture" } } },
        tools: { toolSearch: { enabled: true, mode: "directory" as const } },
      };
      const catalogRef = createToolSearchCatalogRef();
      const controls = createToolSearchTools({ config, catalogRef });
      const existing = {
        ...createStubTool("existing"),
        execute: vi.fn(async () => ({
          content: [{ type: "text" as const, text: "allowed-existing" }],
        })),
      };
      const input = createInput([], [...controls, existing]);
      input.attempt.config = config;
      input.attempt.runtimePlan = undefined;
      Object.assign(input.preparedToolBase, {
        codeModeControlsEnabledForRun: false,
        toolSearchControlsEnabledForRun: true,
        toolSearchConfig: resolveToolSearchConfig(config),
        toolSearchRuntimeConfig: config,
        toolSearchCatalogRef: catalogRef,
        runtimeCapabilityProfile: resolveConversationCapabilityProfile({ config }),
      });
      const deferred = createDeferred<McpToolCatalog>();
      const callTool = vi.fn(async () => ({
        content: [{ type: "text" as const, text: "same-cold-run" }],
      }));
      const runtime: SessionMcpRuntime = {
        sessionId: "session",
        workspaceDir: "/tmp/workspace",
        configFingerprint: "cold",
        createdAt: 0,
        lastUsedAt: 0,
        markUsed() {},
        peekCatalog: () => null,
        getCatalog: () => deferred.promise,
        callTool,
        dispose: async () => {},
        joinCleanup: async () => {},
        acquireLease: () => () => {},
      };
      mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime });
      mocks.materializeBundleMcpToolsForRun.mockImplementation(materializeBundleMcpToolsForRun);
      const bundleTools = await prepareEmbeddedAttemptBundleTools(input);
      expect(bundleTools.bundleMcpRuntime?.tools).toEqual([]);
      const prepared = prepareEmbeddedAttemptToolCatalog({
        attempt: input.attempt,
        setup: input.setup,
        preparedToolBase: input.preparedToolBase,
        bundleTools,
        refreshTools: () => {
          bundleTools.refreshTools();
          prepared.refreshTools();
          refreshSessionTools();
          promptToolPolicy.refresh();
        },
        abortSignal: new AbortController().signal,
        runTrace: createDiagnosticTraceContext(),
        executeCodeModeTool: async () => {
          throw new Error("not Code Mode");
        },
      });
      let activeToolNames = prepared.effectiveTools.map((tool) => tool.name);
      const refreshSessionTools = vi.fn(() => {
        activeToolNames = prepared.effectiveTools.map((tool) => tool.name);
      });
      const promptToolPolicy = createPromptBuildToolPolicy({
        session: {
          getActiveToolNames: () => activeToolNames,
          setActiveToolsByName: (names) => {
            activeToolNames = names;
          },
        },
        effectiveTools: prepared.effectiveTools,
        uncompactedEffectiveTools: bundleTools.uncompactedEffectiveTools,
        tools: bundleTools.tools,
        catalogRef,
        codeModeControlsEnabled: false,
        onApplied: (surface) => {
          prepared.applyPromptToolPolicy(
            new Set([
              ...surface.activeToolNames,
              ...surface.uncompactedEffectiveTools.map((tool) => tool.name),
            ]),
          );
        },
      });
      promptToolPolicy.apply(["existing", ...(allowMcp ? ["probe__query"] : [])]);
      try {
        expect(catalogRef.current?.entries.map((entry) => entry.name)).toEqual(["existing"]);
        const search = prepared.effectiveTools.find((tool) => tool.name === "tool_search")!;
        const searching = search.execute("cold-search", { query: "probe" });
        expect(callTool).not.toHaveBeenCalled();
        deferred.resolve({
          version: 1,
          generatedAt: 1,
          servers: { probe: { serverName: "probe", launchSummary: "probe", toolCount: 1 } },
          tools: [
            {
              serverName: "probe",
              safeServerName: "probe",
              toolName: "query",
              description: "probe",
              inputSchema: { type: "object", properties: {} },
              fallbackDescription: "probe",
            },
          ],
        });
        const result = await searching;
        expect(refreshSessionTools).toHaveBeenCalledTimes(1);
        expect(
          bundleTools.uncompactedEffectiveTools.some((tool) => tool.name === "probe__query"),
        ).toBe(true);
        const call = prepared.effectiveTools.find((tool) => tool.name === "tool_call")!;
        if (allowMcp) {
          expect(JSON.stringify(result)).toContain("probe__query");
          expect(
            JSON.stringify(await call.execute("cold-call", { id: "probe__query", args: {} })),
          ).toContain("same-cold-run");
          expect(callTool).toHaveBeenCalledWith("probe", "query", {});
        } else {
          expect(JSON.stringify(result)).not.toContain("probe__query");
          expect(catalogRef.current?.entries.map((entry) => entry.name)).toEqual(["existing"]);
          await expect(
            call.execute("denied-call", { id: "probe__query", args: {} }),
          ).rejects.toThrow("Unknown tool id");
          expect(callTool).not.toHaveBeenCalled();
        }
        expect(
          JSON.stringify(await call.execute("allowed-call", { id: "existing", args: {} })),
        ).toContain("allowed-existing");
        expect(existing.execute).toHaveBeenCalledTimes(1);
        expect(mocks.materializeBundleMcpToolsForRun).toHaveBeenCalledTimes(1);
      } finally {
        await bundleTools.bundleMcpRuntime?.dispose();
        clearToolSearchCatalog({ catalogRef });
      }
    },
  );

  it.each([
    { allow: ["chrome*"], expected: ["chrome__click"] },
    { allow: ["ch*me*"], expected: ["chrome__click"] },
    { allow: [" CHROME* "], expected: ["chrome__click"] },
    { allow: ["*click"], expected: ["chrome__click", "other__click"] },
    { allow: ["chrome__*"], expected: ["chrome__click"] },
    { allow: ["chrome*."], expected: [] },
    { allow: ["exec*"], expected: [], discover: false },
    { allow: ["chrome"], expected: [], discover: false },
    { allow: [], expected: [], discover: false },
  ])("discovers configured MCP for $allow without widening final tools", async (testCase) => {
    const input = createInput([], []);
    input.attempt.config = {
      plugins: { enabled: false },
      mcp: { servers: { chrome: { command: "unused" }, other: { command: "unused" } } },
    };
    input.attempt.toolsAllow = testCase.allow;
    input.preparedToolBase.effectiveToolsAllow = testCase.allow;
    mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime: {}, releaseLease: () => {} });
    mocks.materializeBundleMcpToolsForRun.mockResolvedValue({
      tools: [{ name: "chrome__click" }, { name: "other__click" }],
    });

    const result = await prepareEmbeddedAttemptBundleTools(input);

    expect(mocks.acquireSessionMcpRuntime).toHaveBeenCalledTimes(
      testCase.discover === false ? 0 : 1,
    );
    expect(result.uncompactedEffectiveTools.map((tool) => tool.name)).toEqual(testCase.expected);
    expect(mocks.createBundleLspToolRuntime).not.toHaveBeenCalled();
  });

  it.each([
    { enabled: false, override: undefined, expected: false },
    { enabled: true, override: false, expected: false },
    { enabled: false, override: true, expected: true },
  ])("uses effective MCP enablement $enabled/$override", async (testCase) => {
    const input = createInput([], []);
    input.attempt.config = {
      plugins: { enabled: false },
      mcp: { servers: { chrome: { command: "unused", enabled: testCase.enabled } } },
    };
    input.attempt.toolsAllow = ["chrome*"];
    if (testCase.override !== undefined) {
      input.attempt.toolOverrides = { mcpServers: { chrome: testCase.override } };
    }

    await prepareEmbeddedAttemptBundleTools(input);

    expect(mocks.acquireSessionMcpRuntime).toHaveBeenCalledTimes(testCase.expected ? 1 : 0);
  });

  it.each([
    { servers: ["chrome dev"], allow: "chrome-dev*" },
    { servers: ["9chrome"], allow: "mcp-9chrome*" },
    { servers: ["chrome dev", "chrome-dev"], allow: "chrome-dev-2*" },
    { servers: ["a".repeat(31), "a".repeat(32)], allow: `${"a".repeat(28)}-2*` },
    { servers: ["bash"], allow: "bash*" },
  ])("uses canonical namespace allocation for $servers", async ({ servers, allow }) => {
    const input = createInput([], []);
    input.attempt.config = {
      plugins: { enabled: false },
      mcp: { servers: Object.fromEntries(servers.map((name) => [name, { command: "unused" }])) },
    };
    input.attempt.toolsAllow = [allow];

    await prepareEmbeddedAttemptBundleTools(input);

    expect(mocks.acquireSessionMcpRuntime).toHaveBeenCalledOnce();
  });

  it.each(["disableTools", "raw", "restart", "model"])(
    "does not discover matching MCP when tools are disabled by %s",
    async (mode) => {
      const input = createInput([], []);
      input.attempt.config = { mcp: { servers: { chrome: { command: "unused" } } } };
      input.attempt.toolsAllow = ["chrome*"];
      input.attempt.disableTools = mode === "disableTools";
      input.isRawModelRun = mode === "raw";
      input.attempt.forceRestartSafeTools = mode === "restart";
      input.preparedToolBase.toolsEnabled = mode !== "model";

      await prepareEmbeddedAttemptBundleTools(input);

      expect(mocks.acquireSessionMcpRuntime).not.toHaveBeenCalled();
    },
  );

  it("allocates configured namespaces after colliding enabled plugin servers", async () => {
    const input = createInput([], []);
    input.attempt.config = {
      plugins: { entries: { "native-mcp": { enabled: true } } },
      mcp: { servers: { "chrome-dev": { command: "unused" } } },
    };
    input.attempt.toolsAllow = ["chrome-dev-2*"];
    input.preparedToolBase.effectiveToolsAllow = input.attempt.toolsAllow;
    const registry = makeRegistry([{ id: "native-mcp", channels: [] }]);
    const record = registry.plugins[0];
    if (!record) {
      throw new Error("missing native plugin fixture");
    }
    record.format = "openclaw";
    record.mcpServers = { "chrome dev": { command: "unused" } };
    const snapshot = createPluginMetadataSnapshot({
      config: input.attempt.config,
      manifestRegistry: registry,
    });
    input.setup.getCurrentAttemptPluginMetadataSnapshot = () => snapshot;
    mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime: {}, releaseLease: () => {} });
    mocks.materializeBundleMcpToolsForRun.mockResolvedValue({
      tools: [{ name: "chrome-dev__click" }, { name: "chrome-dev-2__click" }],
    });

    const result = await prepareEmbeddedAttemptBundleTools(input);

    expect(result.uncompactedEffectiveTools.map((tool) => tool.name)).toEqual([
      "chrome-dev-2__click",
    ]);
  });

  it.each([
    {
      name: "ordinary uncapped runs",
      allow: undefined,
      clients: ["client_read", "client_delete"],
      expected: ["client_read", "client_delete"],
    },
    {
      name: "message-only completion turns",
      allow: ["message"],
      clients: ["client_read", "client_delete"],
      expected: [],
    },
    {
      name: "explicitly empty capabilities",
      allow: [],
      clients: ["client_read"],
      expected: [],
    },
    {
      name: "wildcard capabilities",
      allow: ["*"],
      clients: ["client_read", "client_delete"],
      expected: ["client_read", "client_delete"],
    },
    {
      name: "canonical tool groups",
      allow: ["group:fs"],
      clients: ["read", "write", "exec"],
      expected: ["read", "write"],
    },
    {
      name: "canonical tool aliases",
      allow: ["bash"],
      clients: ["exec", "client_read"],
      expected: ["exec"],
    },
    {
      name: "independent glob intersections",
      allow: attachToolAllowlistIntersection(
        ["client_read", "client_write", "other_read"],
        [["client_*"], ["*_read"]],
      ),
      clients: ["client_read", "client_write", "other_read"],
      expected: ["client_read"],
    },
  ])("applies the effective client-function capability to $name", async (testCase) => {
    const input = createInput([], []);
    const providedClientTools = testCase.clients.map((name) => ({
      type: "function" as const,
      function: { name, parameters: { type: "object" as const } },
    }));
    input.attempt.clientTools = providedClientTools;
    input.attempt.toolsAllow = testCase.allow;
    input.preparedToolBase.effectiveToolsAllow = testCase.allow;

    const result = await prepareEmbeddedAttemptBundleTools(input);

    expect(result.clientTools?.map((tool) => tool.function.name)).toEqual(testCase.expected);
    if (testCase.allow === undefined) {
      expect(result.clientTools).toBe(providedClientTools);
    }
  });

  it("removes unauthorized client names before MCP and LSP tool reservation", async () => {
    const input = createInput([], [{ name: "message" }]);
    input.attempt.toolsAllow = ["client_allowed", "bundle-mcp", "lsp_probe"];
    input.preparedToolBase.effectiveToolsAllow = input.attempt.toolsAllow;
    input.attempt.clientTools = ["client_allowed", "client_forbidden"].map((name) => ({
      type: "function" as const,
      function: { name, parameters: { type: "object" as const } },
    }));
    mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime: {}, releaseLease: () => {} });
    mocks.materializeBundleMcpToolsForRun.mockResolvedValue({ tools: [] });

    const result = await prepareEmbeddedAttemptBundleTools(input);

    expect(result.clientTools?.map((tool) => tool.function.name)).toEqual(["client_allowed"]);
    expect(mocks.materializeBundleMcpToolsForRun).toHaveBeenCalledWith(
      expect.objectContaining({ reservedToolNames: ["message", "client_allowed"] }),
    );
    expect(mocks.createBundleLspToolRuntime).toHaveBeenCalledWith(
      expect.objectContaining({ reservedToolNames: ["message", "client_allowed"] }),
    );
  });

  it("never exposes client functions when the attempt disables every tool", async () => {
    const input = createInput([], []);
    input.attempt.disableTools = true;
    input.attempt.clientTools = [
      {
        type: "function",
        function: { name: "client_forbidden", parameters: { type: "object" } },
      },
    ];

    const result = await prepareEmbeddedAttemptBundleTools(input);

    expect(result.clientTools).toBeUndefined();
    expect(mocks.acquireSessionMcpRuntime).not.toHaveBeenCalled();
    expect(mocks.materializeBundleMcpToolsForRun).not.toHaveBeenCalled();
    expect(mocks.createBundleLspToolRuntime).not.toHaveBeenCalled();
  });

  it("refreshes spawned-child inheritance after authorized MCP tools materialize", async () => {
    const inheritedToolAllowlist = ["sessions_spawn"];
    mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime: {}, releaseLease: () => {} });
    mocks.materializeBundleMcpToolsForRun.mockResolvedValue({
      tools: [{ name: "server__read" }],
    });

    await prepareEmbeddedAttemptBundleTools(
      createInput(inheritedToolAllowlist, [{ name: "sessions_spawn" }]),
    );

    expect(inheritedToolAllowlist).toEqual(["sessions_spawn", "server__read"]);
  });

  it("never adds policy-denied bundled tools to spawned-child inheritance", async () => {
    const inheritedToolAllowlist = ["sessions_spawn"];
    mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime: {}, releaseLease: () => {} });
    mocks.materializeBundleMcpToolsForRun.mockResolvedValue({
      tools: [{ name: "server__read" }, { name: "server__delete" }],
    });
    mocks.applyFinalEffectiveToolPolicy.mockImplementation(
      ({ bundledTools }: { bundledTools: Array<{ name: string }> }) =>
        bundledTools.filter((tool) => tool.name !== "server__delete"),
    );

    await prepareEmbeddedAttemptBundleTools(
      createInput(inheritedToolAllowlist, [{ name: "sessions_spawn" }]),
    );

    expect(inheritedToolAllowlist).toEqual(["sessions_spawn", "server__read"]);
    expect(inheritedToolAllowlist).not.toContain("server__delete");
  });

  it("captures the post-quarantine creator cap with plugin ownership", async () => {
    const coreTool = { name: "automations" };
    const allowedMcpTool = { name: "mail__read" };
    const quarantinedMcpTool = { name: "mail__broken" };
    setPluginToolMeta(allowedMcpTool as never, { pluginId: "bundle-mcp", optional: false });
    setPluginToolMeta(quarantinedMcpTool as never, {
      pluginId: "bundle-mcp",
      optional: false,
    });
    mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime: {}, releaseLease: () => {} });
    mocks.materializeBundleMcpToolsForRun.mockResolvedValue({
      tools: [allowedMcpTool, quarantinedMcpTool],
    });
    mocks.filterRuntimeCompatibleTools.mockImplementation((tools: Array<{ name: string }>) => ({
      tools: tools.filter((tool) => tool.name !== "mail__broken"),
      diagnostics: [{ toolName: "mail__broken", violations: ["unsupported"] }],
    }));
    const input = createInput([], [coreTool]);
    const captureRef: { value?: { version: 1; source: "final-executable-surface" } } = {};
    input.preparedToolBase.cronCreatorToolAllowlistCaptureRef = captureRef;

    await prepareEmbeddedAttemptBundleTools(input);

    expect(input.preparedToolBase.cronCreatorToolAllowlist).toEqual([
      { name: "automations" },
      { name: "mail__read", pluginId: "bundle-mcp" },
    ]);
    expect(captureRef.value).toEqual({
      version: 1,
      source: "final-executable-surface",
    });
  });

  it("refreshes retained tools and capability captures from each schema projection", async () => {
    const { filterRuntimeCompatibleTools } = await vi.importActual<
      typeof import("../../tool-schema-projection.js")
    >("../../tool-schema-projection.js");
    mocks.filterRuntimeCompatibleTools.mockImplementation(filterRuntimeCompatibleTools);
    const first = createStubTool("core_first");
    const bundled = createStubTool("server__read");
    const bundledSchema = { type: "object" };
    bundled.parameters = bundledSchema;
    setPluginToolMeta(bundled, { pluginId: "bundle-mcp", optional: false });
    const core = [first];
    const inherited = ["initial"];
    const input = createInput(inherited, core);
    const creatorTools = input.preparedToolBase.cronCreatorToolAllowlist;
    input.preparedToolBase.cronCreatorToolAllowlistCaptureRef = {};
    mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime: {}, releaseLease: () => {} });
    mocks.materializeBundleMcpToolsForRun.mockResolvedValue({ tools: [bundled] });

    const result = await prepareEmbeddedAttemptBundleTools(input);
    const retained = result.uncompactedEffectiveTools;
    expect(retained.map((tool) => tool.name)).toEqual(["core_first", "server__read"]);
    expect(core).toEqual([first]);

    const second = createStubTool("core_second");
    core.splice(0, core.length, second);
    bundledSchema.type = "array";
    result.refreshTools();

    expect(retained.map((tool) => tool.name)).toEqual(["core_second"]);
    expect(core).toEqual([second]);
    expect(inherited).toEqual(["core_second"]);
    expect(creatorTools).toEqual([{ name: "core_second" }]);

    core.splice(0, core.length, first);
    bundledSchema.type = "object";
    result.refreshTools();

    expect(retained.map((tool) => tool.name)).toEqual(["core_first", "server__read"]);
    expect(core).toEqual([first]);
    expect(inherited).toEqual(["core_first", "server__read"]);
    expect(creatorTools).toEqual([
      { name: "core_first" },
      { name: "server__read", pluginId: "bundle-mcp" },
    ]);
  });

  it.each([undefined, "MCP", "LSP"])(
    "disposes prepared runtimes after policy failure and retains %s cleanup failure",
    async (failedCleanup) => {
      const disposeMcp = vi.fn(async () => {
        if (failedCleanup === "MCP") {
          throw new Error("MCP disposal failed");
        }
      });
      const disposeLsp = vi.fn(async () => {
        if (failedCleanup === "LSP") {
          throw new Error("LSP disposal failed");
        }
      });
      mocks.acquireSessionMcpRuntime.mockResolvedValue({ runtime: {}, releaseLease: () => {} });
      mocks.materializeBundleMcpToolsForRun.mockResolvedValue({
        tools: [],
        dispose: disposeMcp,
      });
      mocks.createBundleLspToolRuntime.mockResolvedValue({
        tools: [],
        dispose: disposeLsp,
      });
      mocks.applyFinalEffectiveToolPolicy.mockImplementation(() => {
        throw new Error("bundle policy failed");
      });

      const input = createInput([], []);

      const cleanupScope = createAgentCleanupScope();
      await expect(
        cleanupScope.run(() => prepareEmbeddedAttemptBundleTools(input)),
      ).rejects.toThrow("bundle policy failed");
      expect(cleanupScope.outcome).toBe(failedCleanup ? "uncertain" : "closed");
      expect(mocks.applyFinalEffectiveToolPolicy).toHaveBeenCalledWith(
        expect.objectContaining({ workspaceDir: "/tmp/workspace" }),
      );
      expect(disposeMcp).toHaveBeenCalledOnce();
      expect(disposeLsp).toHaveBeenCalledOnce();
    },
  );
});
