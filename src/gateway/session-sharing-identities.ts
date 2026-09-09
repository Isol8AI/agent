import type {
  SessionCreatedActor,
  SessionMember,
  SessionMemberEvidence,
  SessionMemberIdentity,
  SessionSharingIdentity,
} from "../../packages/gateway-protocol/src/index.js";
import { loadCombinedSessionStoreForGatewayCore } from "../config/sessions.js";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { listProfiles } from "../state/user-profiles.js";

const UNKNOWN_SHARING_ACTOR_STORAGE_REF = "actor-evidence:unknown";
const UNATTRIBUTED_SHARING_ACTOR_STORAGE_REF = "actor-evidence:unattributed";
const LEGACY_SYNTHETIC_SHARING_ACTOR_STORAGE_REFS = new Set(["local-operator", "operator.admin"]);

export type SharingActorFacts =
  | { state: "present"; actor: SessionSharingIdentity }
  | { state: "unknown" }
  | { state: "absent" };

function sharingIdentityAsMember(identity: SessionSharingIdentity): SessionMemberIdentity | null {
  return identity.type === "human"
    ? { type: "profile", id: identity.id }
    : identity.type === "agent"
      ? { type: "agent", id: identity.id }
      : null;
}

export function sharingActorStorageRef(facts: SharingActorFacts): string {
  return facts.state === "present"
    ? facts.actor.id
    : facts.state === "unknown"
      ? UNKNOWN_SHARING_ACTOR_STORAGE_REF
      : UNATTRIBUTED_SHARING_ACTOR_STORAGE_REF;
}

export function projectSessionMemberEvidence(member: {
  identity: SessionMemberIdentity;
  identityId: string;
  addedBy: string;
  addedAt: number;
}): SessionMemberEvidence {
  const common = {
    identity: member.identity,
    identityId: member.identityId,
    addedAt: member.addedAt,
  };
  if (member.addedBy === UNKNOWN_SHARING_ACTOR_STORAGE_REF) {
    return { ...common, addedByState: "unknown" };
  }
  if (
    member.addedBy === UNATTRIBUTED_SHARING_ACTOR_STORAGE_REF ||
    LEGACY_SYNTHETIC_SHARING_ACTOR_STORAGE_REFS.has(member.addedBy)
  ) {
    return common;
  }
  return { ...common, addedBy: member.addedBy };
}

export function projectLegacySessionMember(member: SessionMemberEvidence): SessionMember | null {
  return member.addedBy
    ? {
        identity: member.identity,
        identityId: member.identityId,
        addedBy: member.addedBy,
        addedAt: member.addedAt,
      }
    : null;
}

export function memberIdentityFromParams(params: {
  identity?: SessionMemberIdentity;
  identityId?: string;
}): SessionMemberIdentity | null {
  const legacyId = params.identityId?.trim();
  const identity = params.identity
    ? { type: params.identity.type, id: params.identity.id.trim() }
    : legacyId
      ? { type: "profile" as const, id: legacyId }
      : undefined;
  return !identity?.id || (legacyId && legacyId !== identity.id) ? null : identity;
}

export function knownSessionIdentities(params: {
  cfg: OpenClawConfig;
  actor: SharingActorFacts;
}): SessionSharingIdentity[] {
  const identities = new Map<string, SessionSharingIdentity>();
  const remember = (identity: SessionCreatedActor | null) => {
    if (!identity?.id) {
      return;
    }
    const current = identities.get(identity.id);
    identities.set(identity.id, {
      type: identity.type,
      id: identity.id,
      ...((identity.label ?? current?.label) ? { label: identity.label ?? current?.label } : {}),
    });
  };
  if (params.actor.state === "present") {
    remember(params.actor.actor);
  }
  const { store } = loadCombinedSessionStoreForGatewayCore(params.cfg, { projection: "list" });
  for (const entry of Object.values(store)) {
    remember(entry.createdActor ?? null);
  }
  for (const profile of listProfiles()) {
    remember({
      type: "human",
      id: profile.id,
      ...(profile.displayName ? { label: profile.displayName } : {}),
    });
  }
  return [...identities.values()];
}

export function isKnownSessionMemberIdentity(
  known: readonly SessionSharingIdentity[],
  member: SessionMemberIdentity,
): boolean {
  return known.some((identity) => {
    const candidate = sharingIdentityAsMember(identity);
    return candidate?.type === member.type && candidate.id === member.id;
  });
}
