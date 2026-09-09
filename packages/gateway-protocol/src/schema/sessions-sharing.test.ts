import { Value } from "typebox/value";
import { describe, expect, it } from "vitest";
import { SESSION_VISIBILITY_VALUES } from "./sessions-sharing-values.js";
import {
  SessionMemberAddParamsSchema,
  SessionMembersListEvidenceResultSchema,
  SessionMembersListResultSchema,
  SessionSharingEvidenceEventSchema,
  SessionSharingEventSchema,
  SessionVisibilitySetParamsSchema,
} from "./sessions-sharing.js";

const baseEvent = {
  action: "visibility",
  sessionKey: "agent:main:main",
  agentId: "main",
  visibility: "draft",
  ts: 1,
} as const;

describe("session sharing protocol", () => {
  it("accepts additive visibility and membership payloads", () => {
    expect(
      Value.Check(SessionVisibilitySetParamsSchema, {
        sessionKey: "agent:main:main",
        visibility: "draft",
      }),
    ).toBe(true);
    expect(
      Value.Check(SessionMemberAddParamsSchema, {
        sessionKey: "agent:main:main",
        identityId: "alice@example.com",
      }),
    ).toBe(true);
    expect(
      Value.Check(SessionMemberAddParamsSchema, {
        sessionKey: "agent:main:main",
        identity: { type: "agent", id: "researcher" },
      }),
    ).toBe(true);
    expect(
      Value.Check(SessionMembersListResultSchema, {
        sessionKey: "agent:main:main",
        members: [
          {
            identity: { type: "profile", id: "alice" },
            identityId: "alice",
            addedBy: "profile-ada",
            addedAt: 1,
          },
        ],
        identities: [],
        role: "owner",
        allowedVisibilities: ["shared", "read-only", "suggest", "draft"],
      }),
    ).toBe(true);
    expect(
      Value.Check(SessionMembersListResultSchema, {
        sessionKey: "agent:main:main",
        members: [
          {
            identity: { type: "profile", id: "bob" },
            identityId: "bob",
            addedByState: "unknown",
            addedAt: 2,
          },
        ],
        identities: [],
        role: "owner",
        allowedVisibilities: [],
      }),
    ).toBe(false);
    for (const member of [
      {
        identity: { type: "profile", id: "alice" },
        identityId: "alice",
        addedBy: "profile-ada",
        addedAt: 1,
      },
      {
        identity: { type: "profile", id: "bob" },
        identityId: "bob",
        addedByState: "unknown",
        addedAt: 2,
      },
      {
        identity: { type: "profile", id: "carol" },
        identityId: "carol",
        addedAt: 3,
      },
    ]) {
      expect(
        Value.Check(SessionMembersListEvidenceResultSchema, {
          sessionKey: "agent:main:main",
          members: [member],
          identities: [],
          role: "owner",
          allowedVisibilities: ["shared", "read-only", "suggest", "draft"],
        }),
      ).toBe(true);
    }
    expect(
      Value.Check(SessionMembersListEvidenceResultSchema, {
        sessionKey: "agent:main:main",
        members: [
          {
            identity: { type: "profile", id: "mixed" },
            identityId: "mixed",
            addedBy: "profile-ada",
            addedByState: "unknown",
            addedAt: 4,
          },
        ],
        identities: [],
        role: "owner",
        allowedVisibilities: [],
      }),
    ).toBe(false);
  });

  it("exposes restricted visibility and closed typed member identities", () => {
    expect(SESSION_VISIBILITY_VALUES).toContain("restricted");
    expect(
      Value.Check(SessionVisibilitySetParamsSchema, {
        sessionKey: "agent:main:private-room",
        visibility: "restricted",
      }),
    ).toBe(true);
    expect(
      Value.Check(SessionMemberAddParamsSchema, {
        sessionKey: "agent:main:private-room",
        identity: { type: "profile", id: "profile-alice", namespace: "forbidden" },
      }),
    ).toBe(false);
  });

  it("rejects unknown visibility modes", () => {
    expect(
      Value.Check(SessionVisibilitySetParamsSchema, {
        sessionKey: "agent:main:main",
        visibility: "private",
      }),
    ).toBe(false);
  });

  it("keeps the legacy event actor required", () => {
    expect(
      Value.Check(SessionSharingEventSchema, {
        ...baseEvent,
        actor: { type: "human", id: "profile-ada", label: "Ada" },
      }),
    ).toBe(true);
    expect(
      Value.Check(SessionSharingEventSchema, {
        ...baseEvent,
        actorState: "unknown",
      }),
    ).toBe(false);
    expect(Value.Check(SessionSharingEventSchema, baseEvent)).toBe(false);
    expect(
      Value.Check(SessionSharingEventSchema, {
        ...baseEvent,
        actor: { type: "human", id: "profile-ada" },
        actorState: "unknown",
      }),
    ).toBe(false);
    expect(
      Value.Check(SessionSharingEvidenceEventSchema, {
        ...baseEvent,
        actorState: "unknown",
      }),
    ).toBe(true);
    expect(Value.Check(SessionSharingEvidenceEventSchema, baseEvent)).toBe(true);
    expect(
      Value.Check(SessionSharingEvidenceEventSchema, {
        ...baseEvent,
        actor: { type: "human", id: "profile-ada" },
      }),
    ).toBe(false);
  });
});
