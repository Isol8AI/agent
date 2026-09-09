import type { OpenClawConfig } from "../../config/types.openclaw.js";
import type { SessionSharingTarget } from "../session-sharing-policy.js";
import { prepareSessionSharing, resolveSessionVisibility } from "../session-sharing.js";
import type { GatewayClient } from "./types.js";

export function canReadChatHistoryTarget(
  cfg: OpenClawConfig,
  client: GatewayClient | null,
  target: SessionSharingTarget,
): boolean {
  const sharing = prepareSessionSharing({ client, cfg });
  return resolveSessionVisibility(target.entry) === "restricted"
    ? sharing.canReadTarget(target)
    : (sharing.entryFilter?.(target.storeKey, target.entry) ?? true);
}
