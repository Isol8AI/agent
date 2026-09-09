import { setImmediate as yieldToEventLoop } from "node:timers/promises";
import { normalizeOptionalString } from "@openclaw/normalization-core/string-coerce";
import { validateSessionsPreviewParams } from "../../../packages/gateway-protocol/src/index.js";
import {
  isSessionPreviewAuthorityCurrent,
  resolveSessionPreviewAuthority,
  type SessionPreviewAuthority,
} from "../session-preview-authority.js";
import { resolveRequestedSessionAgentId } from "../session-request-agent.js";
import { readSessionPreviewItemsFromTranscript } from "../session-transcript-readers.js";
import {
  resolveCanonicalSessionEntryFromStoreKeys,
  resolveGatewaySessionStoreTargetWithStore,
  type SessionsPreviewEntry,
  type SessionsPreviewResult,
} from "../session-utils.js";
import type { GatewayRequestHandler } from "./types.js";
import { assertValidParams } from "./validation.js";

export const sessionsPreviewHandler: GatewayRequestHandler = async ({
  params,
  respond,
  context,
  client,
}) => {
  if (!assertValidParams(params, validateSessionsPreviewParams, "sessions.preview", respond)) {
    return;
  }
  const keys = (Array.isArray(params.keys) ? params.keys : [])
    .map((key) => normalizeOptionalString(key ?? ""))
    .filter((key): key is string => Boolean(key))
    .slice(0, 64);
  const limit = params.limit ?? 12;
  const maxChars = params.maxChars ?? 240;
  if (keys.length === 0) {
    respond(true, { ts: Date.now(), previews: [] } satisfies SessionsPreviewResult, undefined);
    return;
  }

  const cfg = context.getRuntimeConfig();
  const previews: SessionsPreviewEntry[] = [];
  const authorities: Array<{ authority: SessionPreviewAuthority; index: number; key: string }> = [];
  for (const key of keys) {
    if (previews.length > 0) {
      await yieldToEventLoop();
    }
    const requestedAgent = resolveRequestedSessionAgentId(cfg, key);
    if (!requestedAgent.ok) {
      respond(false, undefined, requestedAgent.error);
      return;
    }
    try {
      const target = resolveGatewaySessionStoreTargetWithStore({
        cfg,
        key,
        agentId: requestedAgent.agentId,
        exactRead: true,
        readOnly: true,
      });
      const entry = resolveCanonicalSessionEntryFromStoreKeys(target.store, target.storeKeys);
      const authority = entry?.sessionId
        ? resolveSessionPreviewAuthority({
            cfg,
            client,
            mode: "operator-boundary",
            sessionKey: target.canonicalKey,
            agentId: target.agentId,
          })
        : null;
      if (
        !entry?.sessionId ||
        !authority ||
        authority.sessionId !== entry.sessionId ||
        authority.canonicalKey !== target.canonicalKey ||
        authority.storePath !== target.storePath
      ) {
        previews.push({ key, status: "missing", items: [] });
        continue;
      }
      const items = readSessionPreviewItemsFromTranscript(
        {
          agentId: target.agentId,
          sessionEntry: entry,
          sessionId: entry.sessionId,
          sessionKey: target.canonicalKey,
          storePath: target.storePath,
        },
        limit,
        maxChars,
      );
      authorities.push({ authority, index: previews.length, key });
      previews.push({ key, status: items.length > 0 ? "ok" : "empty", items });
    } catch {
      previews.push({ key, status: "error", items: [] });
    }
  }

  const currentCfg = context.getRuntimeConfig();
  for (const { authority, index, key } of authorities) {
    if (
      !isSessionPreviewAuthorityCurrent({
        authority,
        cfg: currentCfg,
        client,
        mode: "operator-boundary",
      })
    ) {
      previews[index] = { key, status: "missing", items: [] };
    }
  }
  respond(true, { ts: Date.now(), previews } satisfies SessionsPreviewResult, undefined);
};
