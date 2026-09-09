import type { OpenClawConfig } from "../config/types.openclaw.js";
import { hasOperatorBoundary } from "./operator-role-policy.js";
import type { GatewayClient } from "./server-methods/types.js";
import type { SessionSharingTarget } from "./session-sharing-policy.js";
import {
  prepareSessionSharing,
  resolveSessionSharingTarget,
  resolveSessionVisibility,
} from "./session-sharing.js";

export type SessionPreviewAuthority = {
  agentId: string;
  canonicalKey: string;
  sessionId: string;
  storeKey: string;
  storePath: string;
};

type SessionPreviewVisibilityMode = "list" | "operator-boundary";

function canReadPreviewTarget(params: {
  cfg: OpenClawConfig;
  client: GatewayClient | null;
  mode: SessionPreviewVisibilityMode;
  target: SessionSharingTarget;
}): boolean {
  const sharing = prepareSessionSharing({ cfg: params.cfg, client: params.client });
  if (resolveSessionVisibility(params.target.entry) === "restricted") {
    return sharing.canReadTarget(params.target);
  }
  const entryFilter =
    params.mode === "list" || hasOperatorBoundary(params.client, params.cfg)
      ? sharing.entryFilter
      : undefined;
  return entryFilter?.(params.target.storeKey, params.target.entry) ?? true;
}

function captureSessionPreviewAuthority(params: {
  cfg: OpenClawConfig;
  client: GatewayClient | null;
  mode: SessionPreviewVisibilityMode;
  target: SessionSharingTarget;
}): SessionPreviewAuthority | null {
  const sessionId = params.target.entry.sessionId?.trim();
  if (!sessionId || !canReadPreviewTarget(params)) {
    return null;
  }
  return {
    agentId: params.target.agentId,
    canonicalKey: params.target.canonicalKey,
    sessionId,
    storeKey: params.target.storeKey,
    storePath: params.target.storePath,
  };
}

export function resolveSessionPreviewAuthority(params: {
  cfg: OpenClawConfig;
  client: GatewayClient | null;
  mode: SessionPreviewVisibilityMode;
  sessionKey: string;
  agentId?: string;
}): SessionPreviewAuthority | null {
  try {
    const target = resolveSessionSharingTarget({
      cfg: params.cfg,
      sessionKey: params.sessionKey,
      ...(params.agentId ? { agentId: params.agentId } : {}),
      exactRead: true,
    });
    return target ? captureSessionPreviewAuthority({ ...params, target }) : null;
  } catch {
    return null;
  }
}

export function isSessionPreviewAuthorityCurrent(params: {
  authority: SessionPreviewAuthority;
  cfg: OpenClawConfig;
  client: GatewayClient | null;
  mode: SessionPreviewVisibilityMode;
}): boolean {
  const current = resolveSessionPreviewAuthority({
    cfg: params.cfg,
    client: params.client,
    mode: params.mode,
    sessionKey: params.authority.canonicalKey,
    agentId: params.authority.agentId,
  });
  return (
    current !== null &&
    current.agentId === params.authority.agentId &&
    current.canonicalKey === params.authority.canonicalKey &&
    current.storeKey === params.authority.storeKey &&
    current.storePath === params.authority.storePath &&
    current.sessionId === params.authority.sessionId
  );
}
