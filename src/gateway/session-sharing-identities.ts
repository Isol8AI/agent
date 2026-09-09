import type {
  SessionCreatedActor,
  SessionMemberIdentity,
  SessionSharingIdentity,
} from "../../packages/gateway-protocol/src/index.js";
import { loadCombinedSessionStoreForGatewayCore } from "../config/sessions.js";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { listProfiles } from "../state/user-profiles.js";
import { sharingIdentityAsMember } from "./session-sharing-policy.js";

export const UNKNOWN_SHARING_ACTOR_STORAGE_REF = "actor-evidence:unknown";
export const UNATTRIBUTED_SHARING_ACTOR_STORAGE_REF = "actor-evidence:unattributed";

export type SharingActorFacts =
  | { state: "present"; actor: SessionSharingIdentity }
  | { state: "unknown" }
  | { state: "absent" };

export function sharingActorStorageRef(facts: SharingActorFacts): string {
  return facts.state === "present"
    ? facts.actor.id
    : facts.state === "unknown"
      ? UNKNOWN_SHARING_ACTOR_STORAGE_REF
      : UNATTRIBUTED_SHARING_ACTOR_STORAGE_REF;
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
