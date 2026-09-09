import type { OpenClawConfig } from "../../config/types.openclaw.js";
import type { SessionSharingTarget } from "../session-sharing-policy.js";
import { prepareSessionSharing, resolveSessionVisibility } from "../session-sharing.js";
import type { GatewayClient } from "./types.js";

export function prepareChatHistoryAccess(cfg: OpenClawConfig, client: GatewayClient | null) {
  const sharing = prepareSessionSharing({ client, cfg });
  return {
    canReadTarget: (target: SessionSharingTarget) =>
      resolveSessionVisibility(target.entry) === "restricted"
        ? sharing.canReadTarget(target)
        : (sharing.entryFilter?.(target.storeKey, target.entry) ?? true),
    roleForTarget: sharing.roleForTarget,
  };
}
