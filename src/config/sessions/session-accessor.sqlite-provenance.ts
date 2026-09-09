import { executeSqliteQueryTakeFirstSync, getNodeSqliteKysely } from "../../infra/kysely-sync.js";
import type { DB as OpenClawAgentKyselyDatabase } from "../../state/openclaw-agent-db.generated.js";
import type { OpenClawAgentDatabase } from "../../state/openclaw-agent-db.js";
import type { InternalSessionEntry as SessionEntry } from "./types.js";

type SessionProvenanceRow = {
  acp_owned: number;
  hook_external_content_source: "gmail" | "webhook" | null;
  plugin_owner_id: string | null;
  session_entry_provenance: number;
  memory_restricted: number | null;
};

export function bindSessionEntryProvenance(entry: SessionEntry): SessionProvenanceRow {
  const hookSource = entry.hookExternalContentSource;
  // Existing session schemas only admit Gmail/webhook; retain explicit email
  // as generic untrusted provenance instead of dropping its security marker.
  const persistedHookSource = hookSource === "email" ? "webhook" : hookSource;
  return {
    session_entry_provenance: 1,
    memory_restricted:
      entry.visibility === "restricted" || entry.privateRoomExecutionPolicy ? 1 : 0,
    acp_owned: entry.acp ? 1 : 0,
    plugin_owner_id:
      typeof entry.pluginOwnerId === "string" && entry.pluginOwnerId.trim()
        ? entry.pluginOwnerId.trim()
        : null,
    hook_external_content_source:
      persistedHookSource === "gmail" || persistedHookSource === "webhook"
        ? persistedHookSource
        : null,
  };
}

export function resolveSessionEntryProvenanceRow<T extends SessionProvenanceRow>(params: {
  boundSessionRow: T;
  database: OpenClawAgentDatabase;
  entry: SessionEntry;
  previousEntry?: SessionEntry;
}): T {
  const db = getNodeSqliteKysely<OpenClawAgentKyselyDatabase>(params.database.db);
  const existingRoot = executeSqliteQueryTakeFirstSync(
    params.database.db,
    db
      .selectFrom("session_windows")
      .select([
        "session_entry_provenance",
        "memory_restricted",
        "acp_owned",
        "plugin_owner_id",
        "hook_external_content_source",
      ])
      .where("session_id", "=", params.entry.sessionId),
  );
  // Restriction is monotonic for a physical transcript; a later metadata patch cannot declassify it.
  const boundSessionRow = {
    ...params.boundSessionRow,
    memory_restricted:
      existingRoot?.memory_restricted === 1 || params.boundSessionRow.memory_restricted === 1
        ? 1
        : existingRoot
          ? existingRoot.memory_restricted
          : params.boundSessionRow.memory_restricted,
  };
  const hasTranscript = Boolean(
    executeSqliteQueryTakeFirstSync(
      params.database.db,
      db
        .selectFrom("transcript_events")
        .select("seq")
        .where("session_id", "=", params.entry.sessionId)
        .limit(1),
    ),
  );
  // Updates cannot prove provenance for a migrated transcript. Known exclusion metadata is monotonic.
  if (
    existingRoot?.session_entry_provenance === 0 &&
    (params.previousEntry?.sessionId === params.entry.sessionId || hasTranscript)
  ) {
    return {
      ...boundSessionRow,
      session_entry_provenance: 0,
      acp_owned: 0,
      plugin_owner_id: null,
      hook_external_content_source: null,
    };
  }
  return existingRoot?.session_entry_provenance === 1
    ? {
        ...boundSessionRow,
        acp_owned: existingRoot.acp_owned === 1 ? 1 : params.boundSessionRow.acp_owned,
        plugin_owner_id: params.boundSessionRow.plugin_owner_id ?? existingRoot.plugin_owner_id,
        hook_external_content_source:
          params.boundSessionRow.hook_external_content_source ??
          existingRoot.hook_external_content_source,
      }
    : boundSessionRow;
}
