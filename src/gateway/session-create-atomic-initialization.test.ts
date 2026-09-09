import { describe, expect, it } from "vitest";
import {
  loadSessionEntryReadOnly,
  loadTranscriptEvents,
  upsertSessionEntryCore,
} from "../config/sessions/session-accessor.js";
import { listSessionMembers } from "../config/sessions/session-sharing-store.js";
import { withSessionTranscriptWriteLock } from "../plugin-sdk/session-transcript-runtime.js";
import { ensureProfileForEmail } from "../state/user-profiles.js";
import { withOpenClawTestState } from "../test-utils/openclaw-test-state.js";
import { createGatewaySession } from "./session-create-service.js";

async function appendImportedMessage(params: {
  agentId: string;
  sessionId: string;
  sessionKey: string;
  storePath: string;
}) {
  await withSessionTranscriptWriteLock(
    {
      agentId: params.agentId,
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      storePath: params.storePath,
    },
    async (transcript) => {
      await transcript.appendMessage({
        message: { role: "user", content: "Imported snapshot", timestamp: 1 },
      });
    },
  );
}

describe("atomic Gateway session initialization", () => {
  it("creates a restricted thread and its typed ACL as one distinct room node", async () => {
    await withOpenClawTestState({ label: "atomic-restricted-room" }, async () => {
      const owner = ensureProfileForEmail("atomic-room-owner@example.test");
      const profileMember = ensureProfileForEmail("atomic-room-member@example.test");
      const threadProfileMember = ensureProfileForEmail("atomic-thread-member@example.test");
      const creator = {
        via: "operator" as const,
        actor: { type: "human" as const, source: "profile" as const, id: owner.id },
      };
      await upsertSessionEntryCore(
        { agentId: "main", sessionKey: "agent:main:known-agent" },
        {
          sessionId: "known-agent",
          updatedAt: 1,
          createdActor: { type: "agent", id: "same-id" },
        },
      );
      const parent = await createGatewaySession({
        cfg: {},
        key: "agent:main:private-parent",
        commandSource: "test",
        creation: creator,
        visibility: "restricted",
        roomKind: "channel",
        members: [
          { type: "profile", id: profileMember.id },
          { type: "agent", id: "same-id" },
        ],
      });
      expect(parent.ok).toBe(true);
      if (!parent.ok) {
        throw new Error(parent.error.message);
      }

      const thread = await createGatewaySession({
        cfg: {},
        key: "agent:main:private-thread",
        commandSource: "test",
        creation: { via: "spawn", actor: { type: "agent", id: "same-id" } },
        visibility: "restricted",
        roomKind: "thread",
        members: [
          { type: "profile", id: threadProfileMember.id },
          { type: "agent", id: "same-id" },
        ],
        parentSessionKey: parent.key,
        threadOrigin: { parentRoomKey: parent.key, originRootMessageId: "message-root" },
      });
      expect(thread.ok).toBe(true);
      if (!thread.ok) {
        throw new Error(thread.error.message);
      }
      expect(thread.entry.sessionId).not.toBe(parent.entry.sessionId);
      expect(thread.entry).toMatchObject({
        visibility: "restricted",
        roomKind: "thread",
        parentSessionKey: parent.key,
        threadOrigin: { parentRoomKey: parent.key, originRootMessageId: "message-root" },
        sandbox: "required",
      });
      const stored = loadSessionEntryReadOnly({ sessionKey: thread.key });
      expect(stored).toMatchObject({
        privateRoomExecutionPolicy: {
          isolationSubject: { type: "session", sessionId: thread.entry.sessionId },
          sandbox: "required",
          workspaceAccess: "none",
          toolPolicyVersion: "private-room-v1",
          allowedCapabilities: [],
        },
      });
      expect(
        listSessionMembers({ agentId: thread.agentId, sessionKey: thread.key }),
      ).toEqual([
        expect.objectContaining({
          identity: { type: "agent", id: "same-id" },
          addedBy: "same-id",
        }),
        expect.objectContaining({
          identity: { type: "profile", id: threadProfileMember.id },
          addedBy: "same-id",
        }),
      ]);
      expect(listSessionMembers({ agentId: parent.agentId, sessionKey: parent.key })).toEqual([
        expect.objectContaining({
          identity: { type: "agent", id: "same-id" },
          addedBy: owner.id,
        }),
        expect.objectContaining({
          identity: { type: "profile", id: profileMember.id },
          addedBy: owner.id,
        }),
      ]);
      expect(
        await createGatewaySession({
          cfg: {},
          key: "agent:main:phantom-member-room",
          commandSource: "test",
          creation: creator,
          visibility: "restricted",
          roomKind: "group-dm",
          members: [{ type: "profile", id: "phantom-profile" }],
        }),
      ).toMatchObject({
        ok: false,
        error: { message: "unknown restricted room member identity" },
      });
      await expect(
        upsertSessionEntryCore(
          { agentId: thread.agentId, sessionKey: thread.key },
          {
            sessionId: "replacement-session",
            updatedAt: Date.now(),
            visibility: "shared",
            roomKind: "dm",
          },
        ),
      ).rejects.toThrow("Restricted room lifecycle replacement is unavailable");
    });
  });

  it("publishes a usable session only after its transcript initializer succeeds", async () => {
    await withOpenClawTestState({ label: "atomic-session-success" }, async () => {
      let transcriptScope:
        | { agentId: string; sessionId: string; sessionKey: string; storePath: string }
        | undefined;
      const created = await createGatewaySession({
        cfg: {},
        key: "agent:main:atomic-success",
        commandSource: "test",
        operatorRoleActor: { kind: "system" },
        atomicInitialization: true,
        afterCreate: async (entry) => {
          expect(entry.entry.initializationPending).toBe(true);
          transcriptScope = {
            agentId: entry.agentId,
            sessionId: entry.entry.sessionId,
            sessionKey: entry.key,
            storePath: entry.storePath,
          };
          await appendImportedMessage(transcriptScope);
        },
      });

      expect(created).toMatchObject({
        ok: true,
        entry: { initializationPending: undefined },
        postCommit: { status: "completed" },
      });
      if (!created.ok) {
        throw new Error(created.error.message);
      }
      expect(loadSessionEntryReadOnly({ sessionKey: created.key })?.initializationPending).toBe(
        undefined,
      );
      expect(transcriptScope).toBeDefined();
      expect(JSON.stringify(await loadTranscriptEvents(transcriptScope!))).toContain(
        "Imported snapshot",
      );
    });
  });

  it("removes the new session and transcript when initialization fails", async () => {
    await withOpenClawTestState({ label: "atomic-session-failure" }, async () => {
      const sessionKey = "agent:main:atomic-failure";
      const created = await createGatewaySession({
        cfg: {},
        key: sessionKey,
        commandSource: "test",
        operatorRoleActor: { kind: "system" },
        atomicInitialization: true,
        afterCreate: async (entry) => {
          await appendImportedMessage({
            agentId: entry.agentId,
            sessionId: entry.entry.sessionId,
            sessionKey: entry.key,
            storePath: entry.storePath,
          });
          throw new Error("snapshot changed");
        },
      });

      expect(created).toMatchObject({
        ok: false,
        error: { code: "UNAVAILABLE", message: "session initialization failed: snapshot changed" },
      });
      expect(loadSessionEntryReadOnly({ sessionKey })).toBeUndefined();
    });
  });

  it("preserves the existing post-commit contract for ordinary session creation", async () => {
    await withOpenClawTestState({ label: "ordinary-session-initializer" }, async () => {
      const sessionKey = "agent:main:ordinary-initializer";
      const created = await createGatewaySession({
        cfg: {},
        key: sessionKey,
        commandSource: "test",
        operatorRoleActor: { kind: "system" },
        afterCreate: async () => {
          throw new Error("initial turn failed");
        },
      });

      expect(created).toMatchObject({
        ok: true,
        postCommit: { status: "failed", error: expect.any(Error) },
      });
      expect(loadSessionEntryReadOnly({ sessionKey })?.initializationPending).toBeUndefined();
    });
  });
});
