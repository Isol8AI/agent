import {
  ErrorCodes,
  errorShape,
  type ErrorShape,
  type SessionMemberIdentity,
  type SessionSharingRole,
  type SessionVisibility,
} from "../../packages/gateway-protocol/src/index.js";
import { GATEWAY_OWNER_PROFILE_ID } from "../../packages/gateway-protocol/src/schema/users.js";
import {
  isSessionMember,
  type InternalSessionEntry,
  type SessionEntry,
} from "../config/sessions.js";
import { sessionCreatorProfileId } from "../config/sessions/session-entry-provenance.js";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { isIncognitoSessionKey } from "../routing/session-key.js";
import {
  authorizeGatewaySessionCreation,
  operatorSessionCap,
  resolveGatewayOperatorRoleActor,
  resolveOperatorRolePolicy,
} from "./operator-role-policy.js";
import {
  authenticatedProfileUnavailableError,
  gatewayClientSessionCreator,
  isGatewayClientProfilePending,
} from "./server-methods/gateway-client-identity.js";
import type { GatewayClient } from "./server-methods/types.js";
import { prepareSessionCreatorProfile } from "./session-creator.js";
import {
  resolveGatewaySessionStoreTargetsReadOnly,
  type GatewaySessionStoreCache,
  type GatewaySessionStoreDiscoveryCache,
} from "./session-utils-store-lookup.js";
import {
  resolveCanonicalSessionStoreMatchFromStoreKeys,
  resolveGatewaySessionStoreTargetWithStore,
} from "./session-utils.js";

export type SessionSharingTarget = {
  agentId: string;
  canonicalKey: string;
  entry: InternalSessionEntry;
  storeKey: string;
  storeKeys: string[];
  storePath: string;
};

export function resolveSessionVisibility(
  entry: Pick<SessionEntry, "visibility">,
): SessionVisibility {
  return entry.visibility ?? "shared";
}

/** Compare access facts only after the mutation owner has preserved the canonical target. */
export function hasSessionReadAccessChanged(
  previous: SessionEntry | undefined,
  current: SessionEntry,
): boolean {
  return (
    !previous?.sessionId?.trim() ||
    !previous.lifecycleRevision?.trim() ||
    previous.sessionId !== current.sessionId ||
    previous.lifecycleRevision !== current.lifecycleRevision ||
    sessionCreatorProfileId(previous.createdActor) !==
      sessionCreatorProfileId(current.createdActor) ||
    resolveSessionVisibility(previous) !== resolveSessionVisibility(current) ||
    (previous.incognito === true) !== (current.incognito === true)
  );
}

export function isGatewayAdmin(client: Pick<GatewayClient, "connect"> | null): boolean {
  // Internal/plugin-runtime runs reach authorization with a client that has no
  // connect handshake; treat a connect-less client as a non-admin, never a crash.
  return client?.connect?.scopes?.includes("operator.admin") === true;
}

export function allowedSessionVisibilities(cfg: OpenClawConfig): SessionVisibility[] {
  const policy = cfg.session?.sharing;
  return [
    "shared",
    ...(policy?.readOnly === false ? [] : (["read-only"] as const)),
    ...(policy?.suggest === false ? [] : (["suggest"] as const)),
    ...(policy?.drafts === false ? [] : (["draft"] as const)),
  ];
}

export function isSessionVisibilityAllowed(
  cfg: OpenClawConfig,
  visibility: SessionVisibility,
): boolean {
  return visibility === "restricted" || allowedSessionVisibilities(cfg).includes(visibility);
}

export function resolveSessionSharingTarget(params: {
  cfg: OpenClawConfig;
  sessionKey: string;
  agentId?: string;
  exactRead?: boolean;
  storeCache?: GatewaySessionStoreCache;
  targetDiscoveryCache?: GatewaySessionStoreDiscoveryCache;
}): SessionSharingTarget | null {
  const target = resolveGatewaySessionStoreTargetWithStore({
    cfg: params.cfg,
    key: params.sessionKey,
    agentId: params.agentId,
    clone: false,
    // Authorization includes the persisted private-room execution policy.
    projection: "full",
    // Batch callers reuse one store snapshot; single-target checks must not
    // materialize unrelated sessions for every task or authorization recheck.
    exactRead: params.exactRead ?? !params.storeCache,
    ...(params.storeCache ? { storeCache: params.storeCache } : {}),
    ...(params.targetDiscoveryCache ? { targetDiscoveryCache: params.targetDiscoveryCache } : {}),
  });
  return toSessionSharingTarget(target);
}

/** Fresh metadata for one synchronous batch; no authorization decisions are retained. */
export function resolveSessionSharingTargets(params: {
  cfg: OpenClawConfig;
  targets: readonly { sessionKey: string; agentId?: string }[];
}): Array<SessionSharingTarget | null> {
  return resolveGatewaySessionStoreTargetsReadOnly({
    cfg: params.cfg,
    projection: "full",
    targets: params.targets.map(({ sessionKey, agentId }) => ({ key: sessionKey, agentId })),
  }).map(toSessionSharingTarget);
}

function toSessionSharingTarget(
  target: ReturnType<typeof resolveGatewaySessionStoreTargetWithStore>,
): SessionSharingTarget | null {
  const match = resolveCanonicalSessionStoreMatchFromStoreKeys(target.store, target.storeKeys);
  return match
    ? {
        agentId: target.agentId,
        canonicalKey: target.canonicalKey,
        entry: match.entry,
        storeKey: match.key,
        storeKeys: target.storeKeys,
        storePath: target.storePath,
      }
    : null;
}

export type SessionSharingRoleParams = {
  cfg?: OpenClawConfig;
  client: GatewayClient | null;
  target: SessionSharingTarget;
  includeMembership?: boolean;
  isMember?: boolean;
};

export function gatewayClientSessionMemberIdentity(
  client: GatewayClient | null,
  actor: ReturnType<typeof resolveGatewayOperatorRoleActor> =
    resolveGatewayOperatorRoleActor(client),
): SessionMemberIdentity | undefined {
  const profileId =
    gatewayClientSessionCreator(client)?.id ??
    (actor?.kind === "operator" ? actor.profileId : undefined);
  if (profileId) {
    return { type: "profile", id: profileId };
  }
  const agentId = client?.internal?.agentRuntimeIdentity?.agentId?.trim();
  return agentId ? { type: "agent", id: agentId } : undefined;
}

function isSessionCreatorIdentity(
  actor: SessionSharingTarget["entry"]["createdActor"],
  identity: SessionMemberIdentity | undefined,
): boolean {
  return Boolean(
    actor?.id &&
      identity &&
      actor.id === identity.id &&
      ((actor.type === "human" && identity.type === "profile") ||
        (actor.type === "agent" && identity.type === "agent")),
  );
}

export function sharingIdentity(
  client: GatewayClient | null,
  actor: ReturnType<typeof resolveGatewayOperatorRoleActor>,
) {
  const operator = actor?.kind === "operator" ? { id: actor.profileId } : undefined;
  const identity = gatewayClientSessionCreator(client) ?? operator;
  // Owner attribution never narrows sharing; solo deployments stay owner-equivalent.
  return identity?.id === GATEWAY_OWNER_PROFILE_ID ? undefined : identity;
}

export function resolveSessionSharingRole(
  params: SessionSharingRoleParams,
  preparedCap?: { value: ReturnType<typeof operatorSessionCap> },
  isCreator?: ReturnType<typeof prepareSessionCreatorProfile>,
): SessionSharingRole {
  if (isGatewayAdmin(params.client)) {
    return "admin";
  }
  const operatorActor = resolveGatewayOperatorRoleActor(params.client);
  const identity = sharingIdentity(params.client, operatorActor);
  const memberIdentity = gatewayClientSessionMemberIdentity(params.client, operatorActor);
  const visibility = resolveSessionVisibility(params.target.entry);
  if (visibility === "restricted") {
    if (!memberIdentity) {
      return "viewer";
    }
    const creatorMatches =
      isCreator ??
      (memberIdentity.type === "profile"
        ? prepareSessionCreatorProfile(memberIdentity.id)
        : (actor: SessionSharingTarget["entry"]["createdActor"]) =>
            isSessionCreatorIdentity(actor, memberIdentity));
    if (creatorMatches(params.target.entry.createdActor)) {
      return "owner";
    }
    const sessionCap = preparedCap
      ? preparedCap.value
      : params.cfg && operatorSessionCap(params.client, params.cfg);
    if (sessionCap === "none") {
      return "viewer";
    }
    const member =
      params.isMember ??
      (params.includeMembership !== false &&
        isSessionMember(
          {
            agentId: params.target.agentId,
            sessionKey: params.target.storeKey,
            storePath: params.target.storePath,
          },
          memberIdentity,
        ));
    return member ? "member" : "viewer";
  }
  // Solo ownership is independent of the shared-secret connection's attribution profile.
  if (!identity) {
    return params.client?.authenticatedGitHubIdentitySync ||
      (params.cfg?.gateway?.roles && operatorActor?.kind !== "system")
      ? "viewer"
      : "owner";
  }
  const creatorMatches = isCreator ?? prepareSessionCreatorProfile(identity.id);
  if (creatorMatches(params.target.entry.createdActor)) {
    return "owner";
  }
  const sessionCap = preparedCap
    ? preparedCap.value
    : params.cfg && operatorSessionCap(params.client, params.cfg);
  if (
    sessionCap === "write" &&
    visibility !== "draft" &&
    params.target.entry.incognito !== true &&
    !isIncognitoSessionKey(params.target.canonicalKey)
  ) {
    return "member";
  }
  if (sessionCap === "none") {
    return "viewer";
  }
  const member =
    params.isMember ??
    (params.includeMembership !== false &&
      isSessionMember(
        {
          agentId: params.target.agentId,
          sessionKey: params.target.storeKey,
          storePath: params.target.storePath,
        },
        identity.id,
      ));
  return member ? "member" : "viewer";
}

export function canManageSessionSharing(role: SessionSharingRole): boolean {
  return role === "admin" || role === "owner";
}

export function hiddenSessionNotFound(sessionKey: string, incognito = false): ErrorShape {
  const label = incognito ? "Incognito session" : "Session";
  return errorShape(ErrorCodes.INVALID_REQUEST, `${label} "${sessionKey}" was not found.`);
}

function isIncognitoSessionTarget(params: {
  sessionKey: string;
  target: Pick<SessionSharingTarget, "canonicalKey" | "entry"> | null;
}): boolean {
  return params.target
    ? params.target.entry.incognito === true || isIncognitoSessionKey(params.target.canonicalKey)
    : isIncognitoSessionKey(params.sessionKey);
}

export function isResolvedIncognitoSession(params: {
  cfg: OpenClawConfig;
  sessionKey: string;
  agentId?: string;
}): boolean {
  return isIncognitoSessionTarget({
    sessionKey: params.sessionKey,
    target: resolveSessionSharingTarget(params),
  });
}

export function authorizeIncognitoSessionTarget(params: {
  client: GatewayClient | null;
  sessionKey: string;
  target: SessionSharingTarget | null;
}): ErrorShape | null {
  if (!isIncognitoSessionTarget(params)) {
    return null;
  }
  if (isGatewayAdmin(params.client)) {
    return null;
  }
  if (isGatewayClientProfilePending(params.client)) {
    return authenticatedProfileUnavailableError();
  }
  const identity = sharingIdentity(params.client, resolveGatewayOperatorRoleActor(params.client));
  if (!identity) {
    return null;
  }
  return hiddenSessionNotFound(params.sessionKey, true);
}

export function canAccessIncognitoSession(params: {
  cfg: OpenClawConfig;
  client: GatewayClient | null;
  sessionKey: string;
  agentId?: string;
}): boolean {
  if (isGatewayAdmin(params.client)) {
    return true;
  }
  return (
    authorizeIncognitoSessionTarget({
      client: params.client,
      sessionKey: params.sessionKey,
      target: resolveSessionSharingTarget(params),
    }) === null
  );
}

export function authorizeResolvedSessionMutation(params: {
  cfg: OpenClawConfig;
  client: GatewayClient | null;
  sessionKey: string;
  agentId?: string;
}): ErrorShape | null {
  if (isGatewayAdmin(params.client) && !params.cfg.gateway?.roles) {
    return null;
  }
  if (isGatewayClientProfilePending(params.client)) {
    return authenticatedProfileUnavailableError();
  }
  const target = resolveSessionSharingTarget(params);
  if (target) {
    if (resolveSessionVisibility(target.entry) === "restricted") {
      const sharingError = authorizeSessionSharingTarget({
        cfg: params.cfg,
        client: params.client,
        target,
      });
      if (sharingError) {
        return sharingError;
      }
    }
    const agentError = authorizeSessionAgentRun({
      cfg: params.cfg,
      client: params.client,
      target,
    });
    if (agentError) {
      return agentError;
    }
  }
  if (isGatewayAdmin(params.client)) {
    return null;
  }
  const incognitoError = authorizeIncognitoSessionTarget({
    client: params.client,
    sessionKey: params.sessionKey,
    target,
  });
  if (incognitoError) {
    return incognitoError;
  }
  if (!target) {
    return null;
  }
  if (resolveSessionVisibility(target.entry) === "restricted") {
    return null;
  }
  return authorizeSessionSharingTarget({ cfg: params.cfg, client: params.client, target });
}

export function authorizeSessionAgentRun(params: {
  cfg: OpenClawConfig;
  client: GatewayClient | null;
  target: SessionSharingTarget;
}): ErrorShape | null {
  const agentError = authorizeGatewaySessionCreation({
    cfg: params.cfg,
    client: params.client,
    agentId: params.target.agentId,
  });
  if (agentError) {
    return agentError;
  }
  if (resolveSessionVisibility(params.target.entry) === "restricted") {
    return errorShape(
      ErrorCodes.INVALID_REQUEST,
      "private room execution is unavailable until its isolation policy is active",
      {
        details: {
          code: "SESSION_PRIVATE_EXECUTION_UNAVAILABLE",
          sessionKey: params.target.canonicalKey,
        },
      },
    );
  }
  if (
    params.cfg.gateway?.roles &&
    params.target.entry.sandbox !== "required" &&
    resolveOperatorRolePolicy(params.client, params.cfg)?.sandbox === "required"
  ) {
    return errorShape(
      ErrorCodes.FORBIDDEN,
      `Your operator role requires a sandboxed session; create a new session instead of running in "${params.target.canonicalKey}".`,
    );
  }
  return null;
}

export function authorizeSessionSharingTarget(params: {
  cfg?: OpenClawConfig;
  client: GatewayClient | null;
  target: SessionSharingTarget;
}): ErrorShape | null {
  const visibility = resolveSessionVisibility(params.target.entry);
  const sessionCap = params.cfg && operatorSessionCap(params.client, params.cfg);
  const role = resolveSessionSharingRole(params, { value: sessionCap });
  if (sessionCap === "none" && role !== "owner" && role !== "admin") {
    return hiddenSessionNotFound(params.target.canonicalKey);
  }
  const capped = sessionCap === "view" || sessionCap === "suggest";
  // Draft membership is inactive, while an explicit role caps even shared visibility.
  const canMutate =
    visibility === "draft"
      ? canManageSessionSharing(role)
      : role !== "viewer" || (visibility === "shared" && !capped);
  return canMutate
    ? null
    : errorShape(ErrorCodes.INVALID_REQUEST, `session is ${visibility} for this connection`, {
        details: {
          code: "SESSION_PARTICIPATION_REQUIRED",
          sessionKey: params.target.canonicalKey,
          visibility,
        },
      });
}

/** Read authorization preserves public visibility semantics and adds explicit restricted ACLs. */
export function authorizeSessionReadTarget(params: {
  cfg?: OpenClawConfig;
  client: GatewayClient | null;
  target: SessionSharingTarget;
  isMember?: boolean;
}): ErrorShape | null {
  const visibility = resolveSessionVisibility(params.target.entry);
  // Existing visibility modes retain their handler-specific proof-of-knowledge and
  // discovery semantics. This centralized read gate adds only the restricted ACL.
  if (visibility !== "restricted") {
    return null;
  }
  const sessionCap = params.cfg && operatorSessionCap(params.client, params.cfg);
  const role = resolveSessionSharingRole(
    { ...params, isMember: params.isMember },
    { value: sessionCap },
  );
  const readable =
    role === "admin" || role === "owner" || (sessionCap !== "none" && role === "member");
  return readable ? null : hiddenSessionNotFound(params.target.canonicalKey);
}

export function canReadSessionSharingTarget(
  params: Parameters<typeof authorizeSessionReadTarget>[0],
): boolean {
  return authorizeSessionReadTarget(params) === null;
}

export function authorizeSessionSharing(
  params: Parameters<typeof resolveSessionSharingTarget>[0] & { client: GatewayClient | null },
): ErrorShape | null {
  const target = resolveSessionSharingTarget(params);
  return (
    target && authorizeSessionSharingTarget({ cfg: params.cfg, client: params.client, target })
  );
}
