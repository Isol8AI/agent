import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { onTestFinished } from "vitest";
import { replaceSessionEntrySync } from "../config/sessions/session-accessor.entry.js";
import { parseAgentSessionKey } from "../routing/session-key.js";
import {
  closeOpenClawAgentDatabasesForTest,
  openOpenClawAgentDatabase,
} from "../state/openclaw-agent-db.js";
import { AGENT_V14_SESSION_SHARING_SCHEMA_SQL } from "../state/openclaw-agent-session-sharing-schema.js";
import { closeOpenClawStateDatabaseForTest } from "../state/openclaw-state-db.js";
import { SqliteBoardStore } from "./sqlite-board-store.js";

export const v14SharingResetSql = `
  DROP TABLE session_participants;
  DROP TABLE session_members;
  ${AGENT_V14_SESSION_SHARING_SCHEMA_SQL}
`;

export function createTestBoardStore(options: { stateDir?: string } = {}): SqliteBoardStore {
  const ownsStateDir = options.stateDir === undefined;
  const stateDir = options.stateDir ?? mkdtempSync(path.join(tmpdir(), "openclaw-board-store-"));
  const env = { OPENCLAW_STATE_DIR: stateDir };
  const seededSessions = new Set<string>();

  if (ownsStateDir) {
    onTestFinished(() => {
      closeOpenClawAgentDatabasesForTest();
      closeOpenClawStateDatabaseForTest();
      rmSync(stateDir, { recursive: true, force: true });
    });
  }

  return new SqliteBoardStore({
    resolveSession: ({ sessionKey, agentId: requestedAgentId }) => {
      const parsed = parseAgentSessionKey(sessionKey);
      const agentId = requestedAgentId ?? parsed?.agentId ?? "main";
      // Mirror the Gateway resolver so shorthand keys exercise canonical persisted rows.
      const canonicalSessionKey =
        parsed || sessionKey === "global" || sessionKey === "unknown"
          ? sessionKey
          : `agent:${agentId}:${sessionKey}`;
      const identity = `${agentId}\0${canonicalSessionKey}`;
      if (!seededSessions.has(identity)) {
        const database = openOpenClawAgentDatabase({ agentId, env });
        replaceSessionEntrySync(
          { agentId, sessionKey: canonicalSessionKey, storePath: database.path },
          { sessionId: `board-test-${seededSessions.size}`, updatedAt: Date.now() },
        );
        seededSessions.add(identity);
      }
      return { agentId, sessionKey: canonicalSessionKey };
    },
    env,
  });
}
