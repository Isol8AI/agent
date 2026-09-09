import type { OpenClawConfig } from "../config/types.openclaw.js";
import type { NativePresenceAuthority } from "./native-room-presence.js";
import type { GatewayClient } from "./server-methods/types.js";
import { sessionObserverScopeKey } from "./session-observer-model.js";
import {
  authorizeIncognitoSessionTarget,
  canReadSessionSharingTarget,
  resolveSessionSharingTarget,
} from "./session-sharing-policy.js";

/** Bind authenticated human presence to one exact session instance, never to key knowledge. */
export function prepareNativeRoomPresenceAuthority(params: {
  client: GatewayClient | null;
  getConfig: () => OpenClawConfig;
  sessionKey: string;
  agentId?: string;
}): { roomKey: string; authority: NativePresenceAuthority } | undefined {
  const { client } = params;
  const profileId = client?.authenticatedUserProfile?.profileId;
  if (
    !client?.connId ||
    !profileId ||
    client.internal?.syntheticClient ||
    client.invalidated ||
    client.connectionSignal?.aborted
  ) {
    return undefined;
  }
  const target = resolveSessionSharingTarget({
    cfg: params.getConfig(),
    sessionKey: params.sessionKey,
    agentId: params.agentId,
  });
  if (!target) {
    return undefined;
  }
  const sessionId = target.entry.sessionId;
  const authority: NativePresenceAuthority = {
    actor: { type: "profile", id: profileId },
    isAuthorized: () => {
      if (client.invalidated || client.authenticatedUserProfile?.profileId !== profileId) {
        return false;
      }
      const cfg = params.getConfig();
      const current = resolveSessionSharingTarget({
        cfg,
        sessionKey: target.canonicalKey,
        agentId: target.agentId,
      });
      return Boolean(
        current &&
        current.entry.sessionId === sessionId &&
        current.storePath === target.storePath &&
        !authorizeIncognitoSessionTarget({
          client,
          sessionKey: target.canonicalKey,
          target: current,
        }) &&
        canReadSessionSharingTarget({ client, cfg, target: current }),
      );
    },
  };
  return authority.isAuthorized()
    ? { roomKey: sessionObserverScopeKey(target.canonicalKey, target.agentId), authority }
    : undefined;
}
