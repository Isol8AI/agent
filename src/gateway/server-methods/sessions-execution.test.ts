import { Value } from "typebox/value";
import { describe, expect, it, vi } from "vitest";
import { SessionExecutionDispatchParamsSchema } from "../../../packages/gateway-protocol/src/schema/sessions-viewer-presence.js";
import { type SessionMessageAppendResult } from "../../../packages/gateway-protocol/src/schema/sessions.js";
import { PRIVATE_ROOM_CAPABILITIES } from "../../config/sessions/private-room-policy.js";
import {
  createSessionEntryWithTranscript,
  readActiveTranscriptEntryAnchor,
} from "../../config/sessions/session-accessor.js";
import { addSessionMember } from "../../config/sessions/session-sharing-store.js";
import { withOpenClawTestState } from "../../test-utils/openclaw-test-state.js";
import type { AgentTurnIo } from "../agent-turn/types.js";
import {
  registerPrivateRoomExecution,
  revokePrivateRoomExecutions,
} from "../private-room-executions.js";
import { dispatchSessionExecution } from "./sessions-execution.js";
import { appendSessionMessage } from "./sessions-message-append.js";
import { identifiedClient, sessionSharingTestContext } from "./sessions-sharing.test-support.js";
import type { GatewayRequestHandlerOptions } from "./types.js";

const admission = vi.hoisted(() => ({ startTurn: vi.fn() }));
vi.mock("../agent-turn/agent-turn-service.js", () => ({ createAgentTurnService: () => admission }));
vi.mock("../agent-turn/agent-request-preflight.js", () => ({
  prepareAgentRequestPreflight: ({ request }: { request: unknown }) => request,
}));

describe("private execution dispatch boundary", () => {
  const request = {
    sessionKey: "agent:main:room",
    expectedSessionId: "instance",
    inputMessageId: "committed-input",
    idempotencyKey: "dispatch-1",
  };
  it("accepts only a committed input reference and rejects spoofed root, authorship, timestamps, and excessive hops", () => {
    expect(Value.Check(SessionExecutionDispatchParamsSchema, request)).toBe(true);
    for (const extra of [
      { text: "uncommitted" },
      { rootExecutionId: "forged" },
      { senderId: "forged" },
      { timestamp: Date.now() },
      { hopCount: 4 },
    ]) {
      expect(Value.Check(SessionExecutionDispatchParamsSchema, { ...request, ...extra })).toBe(
        false,
      );
    }
  });
  it("denies unauthenticated work before entering admission", async () => {
    const respond = vi.fn();
    await dispatchSessionExecution({
      req: { type: "req", id: "dispatch", method: "sessions.execution.dispatch", params: request },
      params: request,
      respond,
      client: null,
      context: sessionSharingTestContext(vi.fn(), {}),
      isWebchatConnect: () => false,
    } as GatewayRequestHandlerOptions);
    expect(respond).toHaveBeenCalledWith(
      false,
      undefined,
      expect.objectContaining({ message: expect.stringContaining("authenticated") }),
    );
  });
  it("aborts a revoked owner immediately without mutating the committed input", () => {
    const input = Object.freeze({ id: "committed-input", text: "human contribution" });
    const abort = vi.fn();
    let member = true;
    const release = registerPrivateRoomExecution({
      assertCurrent: () => {
        if (!member) {
          throw new Error("revoked");
        }
      },
      abort,
    });
    revokePrivateRoomExecutions();
    expect(abort).not.toHaveBeenCalled();
    member = false;
    revokePrivateRoomExecutions();
    expect(abort).toHaveBeenCalledOnce();
    expect(input.text).toBe("human contribution");
    release();
  });
  it("keeps the canonical human input when execution admission rejects it", async () => {
    await withOpenClawTestState(async () => {
      const scope = {
        agentId: "main",
        sessionKey: request.sessionKey,
        sessionId: request.expectedSessionId,
      };
      await createSessionEntryWithTranscript(scope, () => ({
        ok: true,
        entry: {
          sessionId: scope.sessionId,
          updatedAt: 1,
          visibility: "restricted",
          roomKind: "channel",
          createdActor: { type: "human", source: "profile", id: "member" },
          privateRoomExecutionPolicy: {
            isolationSubject: { type: "session", sessionId: scope.sessionId },
            sandbox: "required",
            workspaceAccess: "none",
            sessionRoot: `/tmp/private-room-fixture/${scope.sessionId}`,
            toolPolicyVersion: "private-room-v1",
            allowedCapabilities: PRIVATE_ROOM_CAPABILITIES,
          },
        },
      }));
      for (const identity of [
        { type: "profile", id: "member" },
        { type: "agent", id: "main" },
      ] as const) {
        addSessionMember(scope, {
          identity,
          addedBy: "member",
          expectedSessionId: scope.sessionId,
        });
      }
      const context = sessionSharingTestContext(vi.fn(), {});
      const client = identifiedClient("member");
      const appended = vi.fn();
      const appendParams = {
        sessionKey: scope.sessionKey,
        expectedSessionId: scope.sessionId,
        idempotencyKey: "input",
        text: "Persist first",
      };
      await appendSessionMessage({
        req: { type: "req", id: "append", method: "sessions.message.append", params: appendParams },
        params: appendParams,
        client,
        context,
        respond: appended,
        isWebchatConnect: () => false,
      });
      expect(appended.mock.calls[0]?.[0]).toBe(true);
      const receipt = appended.mock.calls[0]![1] as SessionMessageAppendResult;
      const dispatchParams = { ...request, inputMessageId: receipt.messageId };
      admission.startTurn.mockImplementationOnce(async ({ io }: { io: AgentTurnIo }) => {
        io.emitAcceptance(
          [false, undefined, { code: "UNAVAILABLE", message: "quota denied" }],
          undefined,
        );
      });
      const responded = vi.fn();
      await dispatchSessionExecution({
        req: {
          type: "req",
          id: "dispatch",
          method: "sessions.execution.dispatch",
          params: dispatchParams,
        },
        params: dispatchParams,
        client,
        context,
        respond: responded,
        isWebchatConnect: () => false,
      });
      expect(admission.startTurn).toHaveBeenCalledOnce();
      expect(responded.mock.calls[0]?.[0]).toBe(false);
      expect(
        readActiveTranscriptEntryAnchor({ ...scope, entryId: receipt.messageId }),
      ).toBeDefined();
    });
  });
});
