import {
  executeSqliteQuerySync,
  executeSqliteQueryTakeFirstSync,
  getNodeSqliteKysely,
} from "../../infra/kysely-sync.js";
import { withOpenClawAgentDatabaseReadOnly } from "../../state/openclaw-agent-db-readonly.js";
import type { DB as OpenClawAgentKyselyDatabase } from "../../state/openclaw-agent-db.generated.js";
import {
  runOpenClawAgentWriteTransaction,
  type OpenClawAgentDatabase,
  type OpenClawAgentDatabaseOptions,
} from "../../state/openclaw-agent-db.js";
import type { SessionMemberIdentity } from "../../../packages/gateway-protocol/src/index.js";
import type { SessionAccessScope } from "./session-accessor.sqlite-contract.js";
import { readSessionEntryInstanceId } from "./session-accessor.sqlite-entry-identity.js";
import { resolveSqliteScope, toDatabaseOptions } from "./session-accessor.sqlite-scope.js";

type SessionMemberDatabase = Pick<OpenClawAgentKyselyDatabase, "session_members">;

type SessionMember = {
  identity: SessionMemberIdentity;
  /** Legacy profile-only alias retained for existing callers. */
  identityId: string;
  addedBy: string;
  addedAt: number;
};

const SESSION_MEMBERSHIP_QUERY_CHUNK_SIZE = 400;

function resolveDatabaseOptions(scope: SessionAccessScope): OpenClawAgentDatabaseOptions {
  return toDatabaseOptions(resolveSqliteScope(scope));
}

function getSessionMemberKysely(database: Pick<OpenClawAgentDatabase, "db">) {
  return getNodeSqliteKysely<SessionMemberDatabase>(database.db);
}

function normalizeMemberIdentity(identity: SessionMemberIdentity | string): SessionMemberIdentity {
  const resolved =
    typeof identity === "string" ? { type: "profile" as const, id: identity.trim() } : identity;
  const id = resolved.id.trim();
  if ((resolved.type !== "profile" && resolved.type !== "agent") || !id) {
    throw new Error("session member identity is required");
  }
  return { type: resolved.type, id };
}

function tryNormalizeMemberIdentity(
  identity: SessionMemberIdentity | string,
): SessionMemberIdentity | undefined {
  try {
    return normalizeMemberIdentity(identity);
  } catch {
    return undefined;
  }
}

function rowMemberIdentity(row: {
  identity_type: string;
  identity_id: string;
}): SessionMemberIdentity {
  if (row.identity_type !== "profile" && row.identity_type !== "agent") {
    throw new Error(`invalid session member identity type: ${row.identity_type}`);
  }
  return { type: row.identity_type, id: row.identity_id };
}

function readSessionMembers<T>(
  scope: SessionAccessScope,
  fallback: T,
  operation: (database: Pick<OpenClawAgentDatabase, "db">) => T,
): T {
  const result = withOpenClawAgentDatabaseReadOnly(operation, resolveDatabaseOptions(scope), {
    throwOnMissingTable: true,
  });
  return result.found ? result.value : fallback;
}

export function listSessionMembers(scope: SessionAccessScope): SessionMember[] {
  return readSessionMembers(scope, [], (database) => {
    const db = getSessionMemberKysely(database);
    return executeSqliteQuerySync(
      database.db,
      db
        .selectFrom("session_members")
        .select(["identity_type", "identity_id", "added_by", "added_at"])
        .where("session_key", "=", resolveSqliteScope(scope).sessionKey)
        .orderBy("identity_type")
        .orderBy("identity_id"),
    ).rows.map((row) => ({
      identity: rowMemberIdentity(row),
      identityId: row.identity_id,
      addedBy: row.added_by,
      addedAt: row.added_at,
    }));
  });
}

export function listSessionMembershipKeys(
  scope: SessionAccessScope,
  sessionKeys: readonly string[],
  identity: SessionMemberIdentity | string,
): Set<string> {
  const normalizedIdentity = tryNormalizeMemberIdentity(identity);
  const normalizedSessionKeys = [...new Set(sessionKeys.map((key) => key.trim()).filter(Boolean))];
  if (!normalizedIdentity || normalizedSessionKeys.length === 0) {
    return new Set();
  }
  return readSessionMembers(scope, new Set<string>(), (database) => {
    const db = getSessionMemberKysely(database);
    const memberships = new Set<string>();
    for (
      let offset = 0;
      offset < normalizedSessionKeys.length;
      offset += SESSION_MEMBERSHIP_QUERY_CHUNK_SIZE
    ) {
      const chunk = normalizedSessionKeys.slice(
        offset,
        offset + SESSION_MEMBERSHIP_QUERY_CHUNK_SIZE,
      );
      const rows = executeSqliteQuerySync(
        database.db,
        db
          .selectFrom("session_members")
          .select("session_key")
          .where("identity_type", "=", normalizedIdentity.type)
          .where("identity_id", "=", normalizedIdentity.id)
          .where("session_key", "in", chunk),
      ).rows;
      for (const row of rows) {
        memberships.add(row.session_key);
      }
    }
    return memberships;
  });
}

export function isSessionMember(
  scope: SessionAccessScope,
  identity: SessionMemberIdentity | string,
): boolean {
  const normalizedIdentity = tryNormalizeMemberIdentity(identity);
  if (!normalizedIdentity) {
    return false;
  }
  return readSessionMembers(scope, false, (database) => {
    const db = getSessionMemberKysely(database);
    return Boolean(
      executeSqliteQueryTakeFirstSync(
        database.db,
        db
          .selectFrom("session_members")
          .select("identity_id")
          .where("session_key", "=", resolveSqliteScope(scope).sessionKey)
          .where("identity_type", "=", normalizedIdentity.type)
          .where("identity_id", "=", normalizedIdentity.id),
      ),
    );
  });
}

// Membership is bound to a live session entry, never a transcript placeholder.
// Authorization is rechecked before these transactions, but a reset/recreate
// can replace the row under the same key in between; the optional expected id
// adds a caller snapshot check after the canonical node/entry check.
function assertAuthorizedSessionInstance(
  database: OpenClawAgentDatabase,
  sessionKey: string,
  expectedSessionId: string | undefined,
): void {
  const sessionId = readSessionEntryInstanceId(database, sessionKey);
  if (
    sessionId === undefined ||
    (expectedSessionId !== undefined && sessionId !== expectedSessionId)
  ) {
    throw new Error("session changed before sharing mutation");
  }
}

export function addSessionMember(
  scope: SessionAccessScope,
  params: {
    identity?: SessionMemberIdentity;
    /** Legacy profile-only input. */
    identityId?: string;
    addedBy: string;
    addedAt?: number;
    expectedSessionId?: string;
  },
): { member: SessionMember; inserted: boolean } {
  const identity = normalizeMemberIdentity(params.identity ?? params.identityId ?? "");
  if (params.identity && params.identityId && params.identityId.trim() !== identity.id) {
    throw new Error("session member identity aliases disagree");
  }
  const addedBy = params.addedBy.trim();
  if (!addedBy) {
    throw new Error("session member identity and actor are required");
  }
  const options = resolveDatabaseOptions(scope);
  const addedAt = params.addedAt ?? Date.now();
  const inserted = runOpenClawAgentWriteTransaction((database) => {
    assertAuthorizedSessionInstance(
      database,
      resolveSqliteScope(scope).sessionKey,
      params.expectedSessionId,
    );
    const db = getSessionMemberKysely(database);
    const result = executeSqliteQuerySync(
      database.db,
      db
        .insertInto("session_members")
        .values({
          session_key: resolveSqliteScope(scope).sessionKey,
          identity_type: identity.type,
          identity_id: identity.id,
          added_by: addedBy,
          added_at: addedAt,
        })
        .onConflict((conflict) =>
          conflict.columns(["session_key", "identity_type", "identity_id"]).doNothing(),
        ),
    );
    return (result.numAffectedRows ?? 0n) > 0n;
  }, options);
  return { member: { identity, identityId: identity.id, addedBy, addedAt }, inserted };
}

/** Seeds a new room ACL inside the same transaction that publishes its session node. */
export function addInitialSessionMembersInTransaction(
  database: OpenClawAgentDatabase,
  scope: SessionAccessScope,
  params: {
    identities: readonly SessionMemberIdentity[];
    addedBy: string;
    addedAt?: number;
    expectedSessionId: string;
  },
): void {
  const sessionKey = resolveSqliteScope(scope).sessionKey;
  assertAuthorizedSessionInstance(database, sessionKey, params.expectedSessionId);
  const addedBy = params.addedBy.trim();
  if (!addedBy) {
    throw new Error("session member actor is required");
  }
  const addedAt = params.addedAt ?? Date.now();
  const db = getSessionMemberKysely(database);
  for (const rawIdentity of params.identities) {
    const identity = normalizeMemberIdentity(rawIdentity);
    executeSqliteQuerySync(
      database.db,
      db
        .insertInto("session_members")
        .values({
          session_key: sessionKey,
          identity_type: identity.type,
          identity_id: identity.id,
          added_by: addedBy,
          added_at: addedAt,
        })
        .onConflict((conflict) =>
          conflict.columns(["session_key", "identity_type", "identity_id"]).doNothing(),
        ),
    );
  }
}

export function removeSessionMember(
  scope: SessionAccessScope,
  identity: SessionMemberIdentity | string,
  expected?: Pick<SessionMember, "addedBy" | "addedAt">,
  expectedSessionId?: string,
): SessionMember | null {
  const normalizedIdentity = tryNormalizeMemberIdentity(identity);
  if (!normalizedIdentity) {
    return null;
  }
  const options = resolveDatabaseOptions(scope);
  return runOpenClawAgentWriteTransaction((database) => {
    assertAuthorizedSessionInstance(
      database,
      resolveSqliteScope(scope).sessionKey,
      expectedSessionId,
    );
    const db = getSessionMemberKysely(database);
    const row = executeSqliteQueryTakeFirstSync(
      database.db,
      db
        .selectFrom("session_members")
        .select(["identity_type", "identity_id", "added_by", "added_at"])
        .where("session_key", "=", resolveSqliteScope(scope).sessionKey)
        .where("identity_type", "=", normalizedIdentity.type)
        .where("identity_id", "=", normalizedIdentity.id),
    );
    if (
      !row ||
      (expected && (row.added_by !== expected.addedBy || row.added_at !== expected.addedAt))
    ) {
      return null;
    }
    executeSqliteQuerySync(
      database.db,
      db
        .deleteFrom("session_members")
        .where("session_key", "=", resolveSqliteScope(scope).sessionKey)
        .where("identity_type", "=", normalizedIdentity.type)
        .where("identity_id", "=", normalizedIdentity.id),
    );
    return {
      identity: rowMemberIdentity(row),
      identityId: row.identity_id,
      addedBy: row.added_by,
      addedAt: row.added_at,
    };
  }, options);
}
