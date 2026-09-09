import { describe, expect, it, vi } from "vitest";
import { withPrivateRoomExecution } from "../agents/private-room-execution.js";
import { createHookRunner } from "./hooks.js";
import { TEST_PLUGIN_AGENT_CTX } from "./hooks.test-fixtures.js";
import { createMockPluginRegistry } from "./hooks.test-helpers.js";

describe("private room completion hook boundary", () => {
  it.each([true, false])(
    "suppresses every agent_end consumer for private completion (success=%s) without affecting ordinary runs",
    async (success) => {
      const capture = vi.fn(async () => {});
      const otherConsumer = vi.fn(async () => {});
      const runner = createHookRunner(
        createMockPluginRegistry([
          { hookName: "agent_end", pluginId: "global-memory", handler: capture, priority: 100 },
          { hookName: "agent_end", pluginId: "another-consumer", handler: otherConsumer },
        ]),
      );
      const event = {
        messages: [{ role: "user", content: "Keep this room text private" }],
        success,
      };
      let active = true;
      await withPrivateRoomExecution(
        {
          agentId: "test-agent",
          rootExecutionId: "private-root",
          runId: "test-run-id",
          hopCount: 0,
          sessionKey: "test-session",
          sessionId: "test-session-id",
          inputMessageId: "committed-private-input",
          assertCurrent: () => {
            if (!active) {
              throw new Error("execution revoked");
            }
          },
          close: () => {
            active = false;
          },
        },
        async () => {
          await Promise.resolve();
          await runner.runAgentEnd(event, TEST_PLUGIN_AGENT_CTX);
          // Teardown does not turn a private transcript into an ordinary completion.
          active = false;
          await runner.runAgentEnd(event, TEST_PLUGIN_AGENT_CTX, { unrefTimeout: true });
        },
      );
      expect(capture).not.toHaveBeenCalled();
      expect(otherConsumer).not.toHaveBeenCalled();

      const ordinaryEvent = {
        messages: [{ role: "user", content: "Ordinary room text" }],
        success,
      };
      await runner.runAgentEnd(ordinaryEvent, TEST_PLUGIN_AGENT_CTX);
      for (const consumer of [capture, otherConsumer]) {
        expect(consumer).toHaveBeenCalledOnce();
        expect(consumer).toHaveBeenCalledWith(
          { ...ordinaryEvent, runId: TEST_PLUGIN_AGENT_CTX.runId },
          TEST_PLUGIN_AGENT_CTX,
        );
      }
    },
  );
});
