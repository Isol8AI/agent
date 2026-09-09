import { asOptionalRecord } from "@openclaw/normalization-core/record-coerce";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  validateSessionMessageAppendParams,
  type SessionMemberIdentity,
  type SessionMessageAppendResult,
} from "../../../packages/gateway-protocol/src/index.js";
import {
  bindPrivateRoomRun,
  withPrivateRoomExecution,
} from "../../agents/private-room-execution.js";
import { readTranscriptSenderIdentity } from "../../chat/sender-identity.js";
import { resolveSessionStorePathCore } from "../../config/sessions/paths.js";
import {
  createSessionEntryWithTranscript,
  deleteSessionEntryLifecycle,
  listSessionParticipantsReadOnly,
  upsertSessionEntryCore,
} from "../../config/sessions/session-accessor.js";
import * as sessionParticipants from "../../config/sessions/session-accessor.sqlite-participants.js";
import {
  loadTranscriptEventsSync,
  readTranscriptEventId,
  readTranscriptEventMessage,
} from "../../config/sessions/session-accessor.sqlite-read.js";
import {
  addSessionMember,
  removeSessionMember,
} from "../../config/sessions/session-sharing-store.js";
import {
  claimAgentRunDelegatedAuthority,
  registerAgentRunContext,
  resetAgentRunRegistryForTest,
  rotateAgentRunRegistryLifecycleGeneration,
} from "../../infra/agent-run-registry.js";
import { onInternalSessionTranscriptUpdate } from "../../sessions/transcript-events.js";
import { withOpenClawTestState } from "../../test-utils/openclaw-test-state.js";
import { createAgentRuntimeApprovalAuthorityValidator } from "../agent-runtime-identity-token.js";
import { projectChatDisplayMessage } from "../chat-display-projection.js";
import { resolveSessionMutationAuthorization } from "../session-sharing.js";
import {
  projectSessionMessagePayload,
  projectTranscriptEntryMessage,
} from "../session-transcript-message.js";
import { appendSessionMessage } from "./sessions-message-append.js";
import {
  identifiedClient,
  sessionSharingTestContext,
  soloClient,
} from "./sessions-sharing.test-support.js";
import type { GatewayClient, GatewayRequestContext, RespondFn } from "./types.js";

const cfg = { agents: { ownership: "explicit" as const, entries: { main: {}, helper: {} } } };
const params = {
  sessionKey: "agent:main:room",
  expectedSessionId: "room-instance",
  idempotencyKey: "contribution-1",
  text: "Hello room",
};
const scope = {
  agentId: "main",
  sessionKey: params.sessionKey,
  sessionId: params.expectedSessionId,
};

afterEach(() => resetAgentRunRegistryForTest());

function context() {
  const result = sessionSharingTestContext(vi.fn(), cfg);
  result.validateAgentRuntimeApprovalAuthority = createAgentRuntimeApprovalAuthorityValidator();
  return result;
}

async function seedRoom(
  target = scope,
  threadOrigin?: { parentRoomKey: string; originRootMessageId: string },
) {
  await createSessionEntryWithTranscript(target, () => ({
    ok: true,
    entry: {
      sessionId: target.sessionId,
      updatedAt: 1,
      visibility: "restricted",
      roomKind: threadOrigin ? "thread" : "channel",
      ...(threadOrigin ? { threadOrigin } : {}),
      createdActor: { type: "human", source: "profile", id: "owner" },
    },
  }));
  for (const identity of [
    { type: "profile", id: "owner" },
    { type: "profile", id: "member" },
    { type: "agent", id: "helper" },
  ] satisfies SessionMemberIdentity[]) {
    addSessionMember(target, { identity, addedBy: "owner", expectedSessionId: target.sessionId });
  }
}

async function invoke(
  patch: Record<string, unknown> = {},
  client: GatewayClient | null = identifiedClient("member"),
  requestContext: GatewayRequestContext = context(),
) {
  const requestParams = { ...params, ...patch };
  const respond = vi.fn<RespondFn>();
  await appendSessionMessage({
    req: { type: "req", id: "append", method: "sessions.message.append", params: requestParams },
    params: requestParams,
    client,
    context: requestContext,
    respond,
    isWebchatConnect: () => false,
  });
  expect(respond).toHaveBeenCalledOnce();
  return respond.mock.calls[0]!;
}

function receipt(result: Awaited<ReturnType<typeof invoke>>): SessionMessageAppendResult {
  expect(result[0]).toBe(true);
  return result[1] as SessionMessageAppendResult;
}

function agentClient(): GatewayClient {
  const operationalRunInstance = { instanceId: "execution-1", runId: "run-1" };
  registerAgentRunContext("run-1", { agentId: "helper", sessionKey: scope.sessionKey });
  const delegatedAuthority = claimAgentRunDelegatedAuthority(operationalRunInstance);
  withPrivateRoomExecution(
    {
      agentId: "helper",
      rootExecutionId: "root-execution-1",
      runId: "run-1",
      hopCount: 0,
      sessionKey: scope.sessionKey,
      sessionId: scope.sessionId,
      inputMessageId: "input-1",
      assertCurrent: () => {},
      close: () => {},
    },
    () => bindPrivateRoomRun(operationalRunInstance),
  );
  return {
    ...soloClient(),
    internal: {
      syntheticClient: true,
      agentRuntimeIdentity: {
        kind: "agentRuntime",
        agentId: "helper",
        sessionKey: scope.sessionKey,
        operationalRunInstance,
        delegatedAuthority: { kind: "local", ...delegatedAuthority },
      },
    },
  };
}

describe("sessions.message.append", () => {
  it.each([
    "senderId",
    "senderIdentity",
    "sender",
    "role",
    "author",
    "displayName",
    "avatar",
    "timestamp",
    "model",
    "deliver",
    "parentId",
  ])("rejects the caller-owned %s field", async (field) => {
    expect(validateSessionMessageAppendParams({ ...params, [field]: "forged" })).toBe(false);
    expect((await invoke({ [field]: "forged" }))[0]).toBe(false);
  });

  it("replays semantically identical mentions regardless of object property order", async () => {
    await withOpenClawTestState({ label: "message-append-mention-order" }, async (state) => {
      await state.writeConfig(cfg);
      await seedRoom();
      const first = receipt(await invoke({ mentions: [{ type: "agent", id: "helper" }] }));
      const replay = receipt(await invoke({ mentions: [{ id: "helper", type: "agent" }] }));
      expect(replay).toEqual({ ...first, appended: false });
      expect(loadTranscriptEventsSync(scope).filter(readTranscriptEventMessage)).toHaveLength(1);
    });
  });

  it("publishes and acknowledges a committed message when participant recording fails", async () => {
    await withOpenClawTestState({ label: "message-append-participant-failure" }, async (state) => {
      await state.writeConfig(cfg);
      await seedRoom();
      const recordParticipant = vi
        .spyOn(sessionParticipants, "recordSessionParticipant")
        .mockImplementation(() => {
          throw new Error("participant projection unavailable");
        });
      const publishedIds: string[] = [];
      const unsubscribe = onInternalSessionTranscriptUpdate((update) => {
        if (update.sessionKey === scope.sessionKey && update.messageId) {
          publishedIds.push(update.messageId);
        }
      });
      try {
        const first = receipt(await invoke());
        expect(first.appended).toBe(true);
        expect(recordParticipant).toHaveBeenCalledOnce();
        expect(publishedIds).toEqual([first.messageId]);
        expect(
          loadTranscriptEventsSync(scope)
            .filter(readTranscriptEventMessage)
            .map(readTranscriptEventId),
        ).toEqual([first.messageId]);
        expect(receipt(await invoke())).toEqual({ ...first, appended: false });
        expect(recordParticipant).toHaveBeenCalledOnce();
        expect(publishedIds).toEqual([first.messageId]);
      } finally {
        unsubscribe();
        recordParticipant.mockRestore();
      }
    });
  });

  it("persists concurrent independent messages on the durable tail, replays receipts, and conflicts on changed content", async () => {
    await withOpenClawTestState({ label: "message-append-order" }, async (state) => {
      await state.writeConfig(cfg);
      await seedRoom();
      const updates: Array<{ messageId: string; storedIds: Array<string | undefined> }> = [];
      const unsubscribe = onInternalSessionTranscriptUpdate((update) => {
        if (update.sessionKey === scope.sessionKey && update.messageId) {
          // Reading inside the event callback must already observe committed storage.
          updates.push({
            messageId: update.messageId,
            storedIds: loadTranscriptEventsSync(scope).map(readTranscriptEventId),
          });
        }
      });
      try {
        const results = await Promise.all([
          invoke(),
          invoke({ idempotencyKey: "contribution-2", text: "Independent message" }),
        ]);
        const ordered = results.map(receipt).toSorted((a, b) => a.messageSeq - b.messageSeq);
        expect(ordered.map((item) => item.messageSeq)).toEqual([1, 2]);
        expect(ordered[1]?.effectiveParentId).toBe(ordered[0]?.messageId);
        const replay = receipt(await invoke());
        expect(replay).toEqual({ ...receipt(results[0]!), appended: false });
        for (const patch of [
          { text: "Changed text" },
          { mentions: [{ type: "agent", id: "helper" }] },
          { replyToId: ordered[1]?.messageId },
        ]) {
          expect((await invoke(patch))[2]?.message).toContain("idempotency conflict");
        }
        expect(updates).toHaveLength(2);
        for (const update of updates) {
          expect(update.storedIds).toContain(update.messageId);
        }
        expect(listSessionParticipantsReadOnly(scope).get(scope.sessionKey)).toEqual([
          expect.objectContaining({
            identity: { type: "profile", id: "member" },
            contributionCount: 2,
          }),
        ]);
      } finally {
        unsubscribe();
      }
    });
  });

  it("rejects missing identities, nonmembers, stale incarnations, and revoked membership at commit", async () => {
    await withOpenClawTestState({ label: "message-append-authority" }, async (state) => {
      await state.writeConfig(cfg);
      await seedRoom();
      for (const client of [null, soloClient(), identifiedClient("stranger")]) {
        expect((await invoke({}, client))[0]).toBe(false);
      }
      expect((await invoke({ expectedSessionId: "old-room-instance" }))[0]).toBe(false);
      const runtimeClient = agentClient();
      const unownedContext = context();
      unownedContext.validateAgentRuntimeApprovalAuthority = undefined;
      expect((await invoke({}, runtimeClient, unownedContext))[0]).toBe(false);
      const pendingAgent = invoke({}, runtimeClient);
      rotateAgentRunRegistryLifecycleGeneration();
      expect((await pendingAgent)[0]).toBe(false);
      const pending = invoke();
      removeSessionMember(scope, { type: "profile", id: "member" }, undefined, scope.sessionId);
      expect((await pending)[0]).toBe(false);
      addSessionMember(scope, {
        identity: { type: "profile", id: "member" },
        addedBy: "owner",
        expectedSessionId: scope.sessionId,
      });
      const broker = identifiedClient("member");
      broker.connect.scopes = ["operator.admin"];
      broker.internal = { trustedHumanBroker: true };
      const pendingBroker = invoke({ idempotencyKey: "broker-contribution" }, broker);
      removeSessionMember(scope, { type: "profile", id: "member" }, undefined, scope.sessionId);
      expect((await pendingBroker)[0]).toBe(false);
      expect(loadTranscriptEventsSync(scope).filter(readTranscriptEventMessage)).toEqual([]);
      expect(listSessionParticipantsReadOnly(scope).size).toBe(0);
      await upsertSessionEntryCore(scope, { sessionId: "replacement", updatedAt: 2 });
      expect((await invoke())[0]).toBe(false);
      await deleteSessionEntryLifecycle({
        agentId: scope.agentId,
        archiveTranscript: false,
        storePath: resolveSessionStorePathCore(undefined, { agentId: scope.agentId }),
        target: { canonicalKey: scope.sessionKey, storeKeys: [scope.sessionKey] },
      });
      expect((await invoke())[0]).toBe(false);
    });
  });

  it("stores semantic replies without branching and rejects parent/child room targets", async () => {
    await withOpenClawTestState({ label: "message-append-reply" }, async (state) => {
      await state.writeConfig(cfg);
      await seedRoom();
      const root = receipt(await invoke());
      const tail = receipt(await invoke({ idempotencyKey: "tail" }));
      const reply = receipt(await invoke({ idempotencyKey: "reply", replyToId: root.messageId }));
      expect(reply.effectiveParentId).toBe(tail.messageId);
      const child = { ...scope, sessionKey: "agent:main:thread", sessionId: "thread-instance" };
      await seedRoom(child, {
        parentRoomKey: scope.sessionKey,
        originRootMessageId: root.messageId,
      });
      expect(
        (
          await invoke({
            sessionKey: child.sessionKey,
            expectedSessionId: child.sessionId,
            replyToId: root.messageId,
          })
        )[0],
      ).toBe(false);
      const childMessage = receipt(
        await invoke({ sessionKey: child.sessionKey, expectedSessionId: child.sessionId }),
      );
      expect(
        (await invoke({ idempotencyKey: "cross-room", replyToId: childMessage.messageId }))[0],
      ).toBe(false);
      const rows = loadTranscriptEventsSync(scope);
      const replyEvent = rows.find((event) => readTranscriptEventId(event) === reply.messageId);
      expect(readTranscriptEventMessage(replyEvent)).toMatchObject({
        __openclaw: { replyToId: root.messageId },
      });
    });
  });

  it.each(["profile", "agent"] as const)(
    "projects stored %s identity and reply metadata identically in live/history",
    async (type) => {
      await withOpenClawTestState({ label: "message-append-projection" }, async (state) => {
        await state.writeConfig(cfg);
        await seedRoom();
        const root = receipt(await invoke());
        const client = type === "agent" ? agentClient() : identifiedClient("member");
        client.internal = {
          ...client.internal,
          senderAttribution: { id: "forged", identity: { type: "profile", id: "forged" } },
        };
        const updates: Parameters<typeof projectSessionMessagePayload>[0][] = [];
        const unsubscribe = onInternalSessionTranscriptUpdate((update) => {
          if (update.sessionKey && update.messageId) {
            updates.push({ ...update, sessionKey: update.sessionKey, message: update.message });
          }
        });
        try {
          const appended = receipt(
            await invoke({ idempotencyKey: "reply", replyToId: root.messageId }, client),
          );
          const event = loadTranscriptEventsSync(scope).find(
            (row) => readTranscriptEventId(row) === appended.messageId,
          );
          const history = projectChatDisplayMessage(
            projectTranscriptEntryMessage(event, appended.messageSeq),
          );
          const live = projectSessionMessagePayload(updates[0]!).payload?.message;
          for (const projection of [history, live]) {
            const senderIdentity = { type, id: type === "agent" ? "helper" : "member" };
            const metadata = asOptionalRecord(asOptionalRecord(projection)?.["__openclaw"]);
            expect(readTranscriptSenderIdentity(metadata?.senderIdentity)).toEqual(senderIdentity);
            expect(projection).toMatchObject({
              timestamp: readTranscriptEventMessage(event)?.timestamp,
              __openclaw: {
                id: appended.messageId,
                seq: appended.messageSeq,
                replyToId: root.messageId,
                senderIdentity,
              },
            });
          }
          if (type === "agent") {
            expect(
              receipt(
                await invoke(
                  {
                    idempotencyKey: "model-changed-key",
                    replyToId: root.messageId,
                  },
                  client,
                ),
              ),
            ).toEqual({ ...appended, appended: false });
            expect(
              (await invoke({ idempotencyKey: "another-key", text: "Another result" }, client))[0],
            ).toBe(false);
            removeSessionMember(scope, { type: "agent", id: "helper" }, undefined, scope.sessionId);
            expect((await invoke({}, client))[0]).toBe(false);
            rotateAgentRunRegistryLifecycleGeneration();
            expect((await invoke({}, client))[0]).toBe(false);
          }
        } finally {
          unsubscribe();
        }
      });
    },
  );

  it("authorizes persistence while restricted-room execution remains unavailable", async () => {
    await withOpenClawTestState({ label: "message-append-policy" }, async (state) => {
      await state.writeConfig(cfg);
      await seedRoom();
      const options = {
        client: identifiedClient("member"),
        context: context(),
        requestParams: params,
      };
      expect(
        resolveSessionMutationAuthorization({ ...options, method: "sessions.message.append" })
          .error,
      ).toBeNull();
      expect(
        resolveSessionMutationAuthorization({ ...options, method: "chat.send" }).error?.details,
      ).toMatchObject({ code: "SESSION_PRIVATE_EXECUTION_UNAVAILABLE" });
    });
  });

  it("admits an authenticated admin with a valid target and commit guard", async () => {
    await withOpenClawTestState({ label: "message-append-admin" }, async (state) => {
      await state.writeConfig(cfg);
      await seedRoom();
      const admin = identifiedClient("admin");
      admin.connect.scopes = ["operator.admin"];
      const authorization = resolveSessionMutationAuthorization({
        client: admin,
        context: context(),
        method: "sessions.message.append",
        requestParams: params,
      });
      expect(authorization).toMatchObject({ error: null, authorization: expect.any(Object) });
      expect(authorization.authorization?.assertCurrent).toEqual(expect.any(Function));
      expect(receipt(await invoke({ idempotencyKey: "admin-contribution" }, admin)).appended).toBe(
        true,
      );
    });
  });

  it.each(["restricted", "draft"] as const)(
    "does not disclose a denied %s room",
    async (visibility) => {
      await withOpenClawTestState(
        { label: `message-append-hidden-${visibility}` },
        async (state) => {
          await state.writeConfig(cfg);
          await seedRoom();
          await upsertSessionEntryCore(scope, { visibility, updatedAt: 2 });
          const outsider = identifiedClient("outsider");
          const expected = `Session "${scope.sessionKey}" was not found.`;
          expect(
            resolveSessionMutationAuthorization({
              client: outsider,
              context: context(),
              method: "sessions.message.append",
              requestParams: params,
            }).error,
          ).toMatchObject({ message: expected });
          const response = await invoke({}, outsider);
          expect(response[2]).toMatchObject({ message: expected });
          expect(JSON.stringify(response[2])).not.toMatch(/restricted|draft|visibility/);
          const missingKey = "agent:main:missing-room";
          expect((await invoke({ sessionKey: missingKey }, outsider))[2]).toMatchObject({
            message: `Session "${missingKey}" was not found.`,
          });
        },
      );
    },
  );
});
