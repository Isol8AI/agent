import { DatabaseSync } from "node:sqlite";
import { describe, expect, it } from "vitest";
import { migrateSessionMembersSchema } from "./openclaw-agent-session-members-migration.js";

describe("session member identity migration", () => {
  it("migrates legacy identity ids into the profile namespace", () => {
    const database = new DatabaseSync(":memory:");
    database.exec(`
      CREATE TABLE session_nodes (session_key TEXT PRIMARY KEY) STRICT;
      INSERT INTO session_nodes (session_key) VALUES ('agent:main:room');
      CREATE TABLE session_members (
        session_key TEXT NOT NULL,
        identity_id TEXT NOT NULL,
        added_by TEXT NOT NULL,
        added_at INTEGER NOT NULL,
        PRIMARY KEY (session_key, identity_id),
        FOREIGN KEY (session_key) REFERENCES session_nodes(session_key) ON DELETE CASCADE
      ) STRICT;
      CREATE INDEX idx_agent_session_members_identity
        ON session_members(identity_id, session_key);
      INSERT INTO session_members (session_key, identity_id, added_by, added_at)
        VALUES ('agent:main:room', 'profile-alice', 'profile-owner', 1);
    `);

    migrateSessionMembersSchema(database, ":memory:");

    expect(
      database
        .prepare("SELECT identity_type, identity_id, added_by, added_at FROM session_members")
        .get(),
    ).toEqual({
      identity_type: "profile",
      identity_id: "profile-alice",
      added_by: "profile-owner",
      added_at: 1,
    });
    database.close();
  });
});
