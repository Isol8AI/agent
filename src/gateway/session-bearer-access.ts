import { isSessionMember } from "../config/sessions.js";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { resolveOperatorRolePolicyForProfile } from "./operator-role-policy.js";
import type { GatewayClient } from "./server-methods/types.js";
import {
  gatewayClientSessionMemberIdentity,
  isGatewayAdmin,
  resolveSessionSharingTarget,
  resolveSessionVisibility,
} from "./session-sharing-policy.js";

export type SessionBearerAccessBinding = {
  sessionId: string;
  reader?: { type: "profile" | "agent"; id: string };
  admin?: true;
};

/** Public sessions need no extra bearer claim; restricted bearers bind their issuing reader. */
export function captureSessionBearerAccess(params: {
  cfg: OpenClawConfig;
  client: GatewayClient | null;
  sessionKey: string;
  agentId?: string;
}): SessionBearerAccessBinding | undefined {
  const target = resolveSessionSharingTarget({
    cfg: params.cfg,
    sessionKey: params.sessionKey,
    ...(params.agentId ? { agentId: params.agentId } : {}),
    exactRead: true,
  });
  if (!target || resolveSessionVisibility(target.entry) !== "restricted") {
    return undefined;
  }
  const reader = gatewayClientSessionMemberIdentity(params.client);
  return {
    sessionId: target.entry.sessionId,
    ...(reader ? { reader } : {}),
    ...(isGatewayAdmin(params.client) ? { admin: true as const } : {}),
  };
}

export function canUseSessionBearerAccess(params: {
  cfg: OpenClawConfig;
  sessionKey: string;
  agentId?: string;
  binding?: SessionBearerAccessBinding;
}): boolean {
  const target = resolveSessionSharingTarget({
    cfg: params.cfg,
    sessionKey: params.sessionKey,
    ...(params.agentId ? { agentId: params.agentId } : {}),
    exactRead: true,
  });
  if (!target) {
    return params.binding === undefined;
  }
  if (resolveSessionVisibility(target.entry) !== "restricted") {
    return true;
  }
  const binding = params.binding;
  if (!binding || binding.sessionId !== target.entry.sessionId) {
    return false;
  }
  if (
    binding.admin === true &&
    (!binding.reader ||
      !params.cfg.gateway?.roles ||
      (binding.reader.type === "profile" &&
        resolveOperatorRolePolicyForProfile(binding.reader.id, params.cfg)?.scopes.includes(
          "operator.admin",
        ) === true))
  ) {
    return true;
  }
  const reader = binding.reader;
  if (!reader) {
    return false;
  }
  const creator = target.entry.createdActor;
  if (
    (reader.type === "profile" && creator?.type === "human" && creator.id === reader.id) ||
    (reader.type === "agent" && creator?.type === "agent" && creator.id === reader.id)
  ) {
    return true;
  }
  return isSessionMember(
    { agentId: target.agentId, sessionKey: target.storeKey, storePath: target.storePath },
    reader,
  );
}

export function canUseSessionBearerCapability(
  params: Parameters<typeof canUseSessionBearerAccess>[0] & { capability: string },
): boolean {
  if (!canUseSessionBearerAccess(params)) {
    return false;
  }
  const target = resolveSessionSharingTarget({
    cfg: params.cfg,
    sessionKey: params.sessionKey,
    ...(params.agentId ? { agentId: params.agentId } : {}),
    exactRead: true,
    projection: "full",
  });
  if (!target || resolveSessionVisibility(target.entry) !== "restricted") {
    return true;
  }
  return (
    target.entry.privateRoomExecutionPolicy?.allowedCapabilities.includes(params.capability) ===
      true ||
    target.entry.privateRoomExecutionPolicy?.allowedCapabilities.includes(
      `gateway:${params.capability}`,
    ) === true
  );
}
