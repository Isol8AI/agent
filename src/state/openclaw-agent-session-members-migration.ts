import type { DatabaseSync } from "node:sqlite";
import { assertSqliteSchemaContains } from "../infra/sqlite-schema-contract.js";
import { tableExists, tableHasColumn } from "./openclaw-state-db-schema-helpers.js";

const CURRENT_SESSION_MEMBERS_SCHEMA = `CREATE TABLE IF NOT EXISTS session_members (
  session_key TEXT NOT NULL,
  identity_type TEXT NOT NULL DEFAULT 'profile' CHECK (identity_type IN ('profile', 'agent')),
  identity_id TEXT NOT NULL,
  added_by TEXT NOT NULL,
  added_at INTEGER NOT NULL,
  PRIMARY KEY (session_key, identity_type, identity_id),
  FOREIGN KEY (session_key) REFERENCES session_nodes(session_key) ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_agent_session_members_identity
  ON session_members(identity_type, identity_id, session_key);`;

const LEGACY_SESSION_MEMBERS_SCHEMA = `CREATE TABLE IF NOT EXISTS session_members (
  session_key TEXT NOT NULL,
  identity_id TEXT NOT NULL,
  added_by TEXT NOT NULL,
  added_at INTEGER NOT NULL,
  PRIMARY KEY (session_key, identity_id),
  FOREIGN KEY (session_key) REFERENCES session_nodes(session_key) ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_agent_session_members_identity
  ON session_members(identity_id, session_key);`;

const MIGRATION_TABLE = "session_members_identity_migration";

/** Historical schema qualification must not require the future identity namespace. */
export function withLegacySessionMembersSchema(sql: string): string {
  return sql.replace(CURRENT_SESSION_MEMBERS_SCHEMA, LEGACY_SESSION_MEMBERS_SCHEMA);
}

export function migrateSessionMembersSchema(database: DatabaseSync, pathname: string): void {
  if (
    !tableExists(database, "session_members") ||
    tableHasColumn(database, "session_members", "identity_type")
  ) {
    return;
  }
  assertSqliteSchemaContains(database, pathname, LEGACY_SESSION_MEMBERS_SCHEMA);
  if (tableExists(database, MIGRATION_TABLE)) {
    throw new Error(`Session member migration table already exists: ${MIGRATION_TABLE}`);
  }
  database.exec("DROP INDEX idx_agent_session_members_identity;");
  const dependencies = database // sqlite-allow-raw -- Inspect historical dependents before rebuilding.
    .prepare(`SELECT name FROM sqlite_schema
      WHERE (type IN ('trigger', 'index') AND tbl_name = 'session_members' AND sql IS NOT NULL)
         OR (type IN ('view', 'trigger') AND sql LIKE '%session_members%')`)
    .all();
  if (dependencies.length > 0) {
    throw new Error("Session member migration cannot rebuild unknown indexes, views, or triggers.");
  }
  for (const table of database // sqlite-allow-raw -- Enumerate historical inbound foreign keys.
    .prepare("SELECT name FROM sqlite_schema WHERE type = 'table'")
    .all()) {
    const foreignKeys = database // sqlite-allow-raw -- PRAGMA requires a quoted database-owned identifier.
      .prepare(`PRAGMA foreign_key_list("${String(table.name).replaceAll('"', '""')}")`)
      .all();
    if (foreignKeys.some((key) => key.table === "session_members")) {
      throw new Error(
        "Session member migration cannot rebuild a table referenced by an unknown foreign key.",
      );
    }
  }
  database.exec(/* sqlite-allow-raw -- Versioned table rebuild inside the maintenance transaction. */ `
    ${CURRENT_SESSION_MEMBERS_SCHEMA.split("\n\nCREATE INDEX")[0]!.replace(
      "IF NOT EXISTS session_members",
      MIGRATION_TABLE,
    )}
    INSERT INTO ${MIGRATION_TABLE}
      (session_key, identity_type, identity_id, added_by, added_at)
    SELECT session_key, 'profile', identity_id, added_by, added_at
    FROM session_members;
    DROP TABLE session_members;
    ALTER TABLE ${MIGRATION_TABLE} RENAME TO session_members;
  `);
}
