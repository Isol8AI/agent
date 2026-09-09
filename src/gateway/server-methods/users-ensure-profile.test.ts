import { afterEach, expect, test } from "vitest";
import {
  GATEWAY_OWNER_PROFILE_ID,
  validateUsersEnsureProfileResult,
} from "../../../packages/gateway-protocol/src/index.js";
import { closeOpenClawStateDatabaseForTest } from "../../state/openclaw-state-db.js";
import {
  ensureGatewayOwnerProfile,
  getUserProfileListItem,
  listProfiles,
} from "../../state/user-profiles.js";
import { createOpenClawTestState } from "../../test-utils/openclaw-test-state.js";
import { usersHandlers } from "./users.js";

const ALIAS_A = `clerk-${"a".repeat(64)}@identity.lightbulb.invalid`;
const ALIAS_B = `clerk-${"b".repeat(64)}@identity.lightbulb.invalid`;

async function ensureProfile(params: Record<string, unknown>) {
  let result: { ok: boolean; payload?: unknown; error?: unknown } | undefined;
  await usersHandlers["users.ensureProfile"]!({
    req: {} as never,
    params,
    respond: (ok, payload, error) => {
      result = { ok, payload, error };
    },
    context: {} as never,
    client: { connect: { scopes: ["operator.admin"] } } as never,
    isWebchatConnect: () => false,
  });
  return result;
}

afterEach(() => {
  closeOpenClawStateDatabaseForTest();
});

test("users.ensureProfile idempotently creates distinct canonical issuer aliases", async () => {
  const state = await createOpenClawTestState({
    layout: "state-only",
    prefix: "users-ensure-profile-",
  });
  try {
    const owner = ensureGatewayOwnerProfile("Gateway Owner");
    const ownerBefore = getUserProfileListItem(owner.id);

    const first = await ensureProfile({ alias: ALIAS_A });
    const repeated = await ensureProfile({ alias: ALIAS_A });
    const distinct = await ensureProfile({ alias: ALIAS_B });

    const firstPayload = first?.payload;
    const repeatedPayload = repeated?.payload;
    const distinctPayload = distinct?.payload;
    expect(first?.ok).toBe(true);
    expect(validateUsersEnsureProfileResult(firstPayload)).toBe(true);
    expect(validateUsersEnsureProfileResult(repeatedPayload)).toBe(true);
    expect(validateUsersEnsureProfileResult(distinctPayload)).toBe(true);
    if (
      !validateUsersEnsureProfileResult(firstPayload) ||
      !validateUsersEnsureProfileResult(repeatedPayload) ||
      !validateUsersEnsureProfileResult(distinctPayload)
    ) {
      throw new Error("users.ensureProfile returned an invalid result");
    }
    const firstId = firstPayload.profile.id;
    const repeatedId = repeatedPayload.profile.id;
    const distinctId = distinctPayload.profile.id;
    expect(repeatedId).toBe(firstId);
    expect(distinctId).not.toBe(firstId);
    expect([firstId, distinctId]).not.toContain(GATEWAY_OWNER_PROFILE_ID);
    expect(getUserProfileListItem(GATEWAY_OWNER_PROFILE_ID)).toEqual(ownerBefore);
  } finally {
    await state.cleanup();
  }
});

test("users.ensureProfile rejects ordinary aliases and caller-selected identity fields", async () => {
  const state = await createOpenClawTestState({
    layout: "state-only",
    prefix: "users-ensure-profile-invalid-",
  });
  try {
    for (const params of [
      { alias: "person@example.com" },
      { alias: ALIAS_A.toUpperCase() },
      { alias: ` ${ALIAS_A}` },
      { alias: ALIAS_A, profileId: "gateway-owner" },
      { alias: ALIAS_A, displayName: "Administrator" },
      { alias: ALIAS_A, role: "admin", scopes: ["operator.admin"] },
    ]) {
      expect(await ensureProfile(params)).toMatchObject({
        ok: false,
        error: { code: "INVALID_REQUEST" },
      });
    }
    expect(listProfiles()).toEqual([]);
  } finally {
    await state.cleanup();
  }
});
