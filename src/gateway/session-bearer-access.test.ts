import { afterEach, expect, test } from "vitest";
import { upsertSessionEntryCore } from "../config/sessions/session-accessor.js";
import { addSessionMember, removeSessionMember } from "../config/sessions/session-sharing-store.js";
import { closeOpenClawAgentDatabasesForTest } from "../state/openclaw-agent-db.js";
import { ensureProfileForEmail } from "../state/user-profiles.js";
import { withOpenClawTestState } from "../test-utils/openclaw-test-state.js";
import {
  canUseSessionBearerAccess,
  captureSessionBearerAccess,
} from "./session-bearer-access.js";
import { sharingPolicyClient } from "./session-sharing.test-utils.js";

afterEach(() => closeOpenClawAgentDatabasesForTest());

test("restricted bearer access follows durable typed membership revocation", async () => {
  await withOpenClawTestState({ scenario: "minimal" }, async () => {
    const owner = ensureProfileForEmail("bearer-owner@example.test");
    const member = ensureProfileForEmail("bearer-member@example.test");
    const sessionKey = "agent:main:restricted-bearer";
    const sessionId = "session-restricted-bearer";
    await upsertSessionEntryCore(
      { agentId: "main", sessionKey },
      {
        sessionId,
        updatedAt: 1,
        visibility: "restricted",
        sandbox: "required",
        createdActor: { type: "human", source: "profile", id: owner.id },
      },
    );
    addSessionMember(
      { agentId: "main", sessionKey },
      {
        identity: { type: "profile", id: member.id },
        addedBy: owner.id,
        expectedSessionId: sessionId,
      },
    );
    const cfg = {};
    const binding = captureSessionBearerAccess({
      cfg,
      client: sharingPolicyClient({ user: member.id }),
      sessionKey,
    });

    expect(binding).toBeDefined();
    expect(canUseSessionBearerAccess({ cfg, sessionKey, ...(binding ? { binding } : {}) })).toBe(
      true,
    );

    removeSessionMember(
      { agentId: "main", sessionKey },
      { type: "profile", id: member.id },
      undefined,
      sessionId,
    );
    expect(canUseSessionBearerAccess({ cfg, sessionKey, ...(binding ? { binding } : {}) })).toBe(
      false,
    );
  });
});
