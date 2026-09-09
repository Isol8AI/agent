import { listSessionMembershipKeys, type SessionEntry } from "../config/sessions.js";
import type { GatewayStoredSessionTargets } from "../config/sessions/combined-store-gateway.js";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import type { GatewayClient } from "./server-methods/types.js";
import {
  gatewayClientSessionMemberIdentity,
  isGatewayAdmin,
  prepareSessionSharing,
  resolveSessionVisibility,
} from "./session-sharing.js";

/**
 * Builds one caller-specific store filter before any list search, facet, paging,
 * transcript, or child-link projection can observe a restricted row.
 */
export function createAuthorizedSessionListEntryFilter(params: {
  cfg: OpenClawConfig;
  client: GatewayClient | null;
  store: Record<string, SessionEntry>;
  targetsBySessionKey: GatewayStoredSessionTargets;
}): ((key: string, entry: SessionEntry) => boolean) | undefined {
  const sharing = prepareSessionSharing({ client: params.client, cfg: params.cfg });
  const ordinaryFilter = sharing.entryFilter;
  if (isGatewayAdmin(params.client)) {
    return ordinaryFilter;
  }

  const restrictedTargets = Object.entries(params.store).flatMap(([key, entry]) => {
    if (resolveSessionVisibility(entry) !== "restricted") {
      return [];
    }
    const owner = params.targetsBySessionKey.get(key);
    if (!owner) {
      return [];
    }
    const storeKey = owner.storeKey ?? key;
    return [
      {
        key,
        target: {
          agentId: owner.agentId,
          canonicalKey: storeKey,
          entry,
          storeKey,
          storeKeys: [storeKey],
          storePath: owner.storeTarget.storePath,
          storeTarget: owner.storeTarget,
        },
      },
    ];
  });
  if (restrictedTargets.length === 0) {
    return ordinaryFilter;
  }

  const membershipKeys = new Set<string>();
  const memberIdentity = gatewayClientSessionMemberIdentity(params.client);
  if (memberIdentity) {
    const groups = new Map<
      string,
      { agentId: string; sessionKeys: string[]; storePath: string }
    >();
    for (const { target } of restrictedTargets) {
      const identity = `${target.storeTarget.agentId}\0${target.storePath}`;
      const group = groups.get(identity) ?? {
        agentId: target.storeTarget.agentId,
        sessionKeys: [],
        storePath: target.storePath,
      };
      group.sessionKeys.push(target.storeKey);
      groups.set(identity, group);
    }
    for (const group of groups.values()) {
      const firstSessionKey = group.sessionKeys[0];
      if (!firstSessionKey) {
        continue;
      }
      for (const sessionKey of listSessionMembershipKeys(
        {
          agentId: group.agentId,
          sessionKey: firstSessionKey,
          storePath: group.storePath,
        },
        group.sessionKeys,
        memberIdentity,
      )) {
        membershipKeys.add(`${group.agentId}\0${group.storePath}\0${sessionKey}`);
      }
    }
  }

  const authorizedRestrictedKeys = new Set(
    restrictedTargets.flatMap(({ key, target }) =>
      sharing.canReadTarget(
        target,
        membershipKeys.has(
          `${target.storeTarget.agentId}\0${target.storePath}\0${target.storeKey}`,
        ),
      )
        ? [key]
        : [],
    ),
  );
  return (key, entry) =>
    resolveSessionVisibility(entry) === "restricted"
      ? authorizedRestrictedKeys.has(key)
      : (ordinaryFilter?.(key, entry) ?? true);
}
