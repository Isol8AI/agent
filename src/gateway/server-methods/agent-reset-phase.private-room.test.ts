import { describe, expect, it, vi } from "vitest";
import { withPrivateRoomExecution } from "../../agents/private-room-execution.js";
import { runAgentResetPhase } from "./agent-reset-phase.js";
import { sessionSharingTestContext, soloClient } from "./sessions-sharing.test-support.js";

describe("committed private room input", () => {
  it.each(["/reset", "/new", "/reset follow-up"])(
    "keeps %s literal without entering reset lifecycle effects",
    async (message) => {
      const abortForLifecycleRotation = vi.fn(() => true);
      const respond = vi.fn();
      const setCommittedResetCompletion = vi.fn();
      const params = {
        request: { message, idempotencyKey: "run" },
        cfg: {},
        requestedSessionKey: "agent:main:room",
        resolvedSessionId: "room-instance",
        effectiveTranscriptInputText: message,
        message,
        lifecycleGeneration: "generation",
        runId: "run",
        agentDedupeKeys: [],
        client: soloClient(),
        context: sessionSharingTestContext(vi.fn()),
        respond,
        abortForLifecycleRotation,
        setCommittedResetCompletion,
      };
      const result = await withPrivateRoomExecution(
        {
          agentId: "main",
          rootExecutionId: "root",
          runId: "run",
          hopCount: 0,
          sessionKey: "agent:main:room",
          sessionId: "room-instance",
          inputMessageId: "committed-input",
          assertCurrent: () => {},
          close: () => {},
        },
        () => runAgentResetPhase(params),
      );
      expect(result).toEqual({
        stop: false,
        accepted: false,
        requestedSessionKey: "agent:main:room",
        resolvedSessionId: "room-instance",
        effectiveTranscriptInputText: message,
        message,
      });
      expect(abortForLifecycleRotation).not.toHaveBeenCalled();
      expect(respond).not.toHaveBeenCalled();
      expect(setCommittedResetCompletion).not.toHaveBeenCalled();

      // The same ordinary chat input still enters its existing reset lifecycle.
      expect(await runAgentResetPhase(params)).toMatchObject({ stop: true, accepted: true });
      expect(abortForLifecycleRotation).toHaveBeenCalledOnce();
    },
  );
});
