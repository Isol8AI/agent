import { normalizeOptionalString } from "@openclaw/normalization-core/string-coerce";
import { validateSessionsDescribeParams } from "../../../packages/gateway-protocol/src/index.js";
import type { InternalSessionEntry } from "../../config/sessions.js";
import { hasOperatorBoundary } from "../operator-role-policy.js";
import { resolveRequestedSessionAgentId } from "../session-request-agent.js";
import {
  createSessionListEntryFilter,
  prepareSessionSharing,
  resolveSessionVisibility,
} from "../session-sharing.js";
import { readRecentSessionMessagesWithStatsAsync } from "../session-transcript-readers.js";
import { buildSessionListRowMetadataContext } from "../session-utils-projection.js";
import { buildGatewaySessionRow } from "../session-utils.js";
import { readSessionPlacementFields } from "./session-placement-read-projection.js";
import { loadSessionEntriesForTarget, requireSessionKey } from "./sessions-shared.js";
import type { GatewayRequestHandlers } from "./types.js";
import { assertValidParams } from "./validation.js";

function createRoleVisibilityFilter(
  client: Parameters<typeof hasOperatorBoundary>[0],
  cfg: Parameters<typeof hasOperatorBoundary>[1],
) {
  return hasOperatorBoundary(client, cfg)
    ? createSessionListEntryFilter({ client, cfg })
    : undefined;
}

function canReadLoadedEntry(params: {
  client: Parameters<typeof hasOperatorBoundary>[0];
  cfg: Parameters<typeof hasOperatorBoundary>[1];
  entry: InternalSessionEntry;
  target: { agentId: string; canonicalKey: string; storeKeys: string[] };
  storePath: string;
}): boolean {
  if (resolveSessionVisibility(params.entry) !== "restricted") {
    return createRoleVisibilityFilter(params.client, params.cfg)?.(
      params.target.canonicalKey,
      params.entry,
    ) ?? true;
  }
  return prepareSessionSharing({ client: params.client, cfg: params.cfg }).canReadTarget({
    agentId: params.target.agentId,
    canonicalKey: params.target.canonicalKey,
    entry: params.entry,
    storeKey: params.target.canonicalKey,
    storeKeys: params.target.storeKeys,
    storePath: params.storePath,
  });
}

function filterReadableChildSessions(params: {
  childSessions: string[] | undefined;
  client: Parameters<typeof hasOperatorBoundary>[0];
  cfg: Parameters<typeof hasOperatorBoundary>[1];
}): string[] | undefined {
  const readable = params.childSessions?.filter((key) => {
    const requestedAgent = resolveRequestedSessionAgentId(params.cfg, key);
    if (!requestedAgent.ok) {
      return false;
    }
    const child = loadSessionEntriesForTarget({
      key,
      cfg: params.cfg,
      agentId: requestedAgent.agentId,
    });
    return Boolean(
      child.entry &&
        canReadLoadedEntry({
          client: params.client,
          cfg: params.cfg,
          entry: child.entry,
          target: child.target,
          storePath: child.storePath,
        }),
    );
  });
  return readable?.length ? readable : undefined;
}

export const sessionByKeyReadHandlers: GatewayRequestHandlers = {
  "sessions.describe": ({ params, respond, context, client }) => {
    if (!assertValidParams(params, validateSessionsDescribeParams, "sessions.describe", respond)) {
      return;
    }
    const key = requireSessionKey(params.key, respond);
    if (!key) {
      return;
    }
    const cfg = context.getRuntimeConfig();
    const requestedAgent = resolveRequestedSessionAgentId(cfg, key, params.agentId);
    if (!requestedAgent.ok) {
      respond(false, undefined, requestedAgent.error);
      return;
    }
    const { target, storePath, store, entry } = loadSessionEntriesForTarget({
      key,
      cfg,
      includeStoreChildEntries: true,
      ...(requestedAgent.agentId ? { agentId: requestedAgent.agentId } : {}),
    });
    if (!entry || !canReadLoadedEntry({ client, cfg, entry, target, storePath })) {
      respond(true, { session: null }, undefined);
      return;
    }
    const row = buildGatewaySessionRow({
      cfg,
      storePath,
      store,
      key: target.canonicalKey,
      entry,
      agentId: target.agentId,
      includeDerivedTitles: params.includeDerivedTitles,
      includeLastMessage: params.includeLastMessage,
      transcriptUsageMaxBytes: 64 * 1024,
      rowContext: buildSessionListRowMetadataContext({ now: Date.now() }),
      includeSwarmChildren: true,
    });
    const childSessions = filterReadableChildSessions({
      childSessions: row.childSessions,
      client,
      cfg,
    });
    if (childSessions) {
      row.childSessions = childSessions;
    } else {
      delete row.childSessions;
    }
    Object.assign(row, readSessionPlacementFields(context, row.sessionId));
    respond(true, { session: row });
  },
  "sessions.get": async ({ params, respond, context, client }) => {
    // SAFETY: Gateway dispatch supplies object params; each optional field is narrowed before use.
    const p = params as {
      key?: unknown;
      sessionKey?: unknown;
      limit?: unknown;
      agentId?: unknown;
    };
    const key = requireSessionKey(p.key ?? p.sessionKey, respond);
    if (!key) {
      return;
    }
    const limit =
      typeof p.limit === "number" && Number.isFinite(p.limit)
        ? Math.max(1, Math.floor(p.limit))
        : 200;

    const cfg = context.getRuntimeConfig();
    const requestedAgent = resolveRequestedSessionAgentId(
      cfg,
      key,
      normalizeOptionalString(p.agentId),
    );
    if (!requestedAgent.ok) {
      respond(false, undefined, requestedAgent.error);
      return;
    }
    const { target, storePath, entry } = loadSessionEntriesForTarget({
      key,
      cfg,
      agentId: requestedAgent.agentId,
    });
    if (!entry?.sessionId || !canReadLoadedEntry({ client, cfg, entry, target, storePath })) {
      respond(true, { messages: [] }, undefined);
      return;
    }
    const sessionId = entry.sessionId;
    const { messages } = await readRecentSessionMessagesWithStatsAsync(
      {
        agentId: target.agentId,
        sessionEntry: entry,
        sessionId,
        sessionKey: target.canonicalKey,
        storePath,
      },
      {
        maxMessages: limit,
        maxLines: limit * 20 + 20,
        allowResetArchiveFallback: true,
      },
    );
    const currentCfg = context.getRuntimeConfig();
    const currentRequestedAgent = resolveRequestedSessionAgentId(
      currentCfg,
      key,
      normalizeOptionalString(p.agentId),
    );
    const current = currentRequestedAgent.ok
      ? loadSessionEntriesForTarget({
          key,
          cfg: currentCfg,
          agentId: currentRequestedAgent.agentId,
        })
      : null;
    if (
      !current ||
      current.target.agentId !== target.agentId ||
      current.target.canonicalKey !== target.canonicalKey ||
      current.storePath !== storePath ||
      !current.entry ||
      current.entry.sessionId !== sessionId ||
      !canReadLoadedEntry({
        client,
        cfg: currentCfg,
        entry: current.entry,
        target: current.target,
        storePath: current.storePath,
      })
    ) {
      respond(true, { messages: [] }, undefined);
      return;
    }
    respond(true, { messages }, undefined);
  },
};
