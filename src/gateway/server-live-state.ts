// Gateway live state factory.
// Combines mutable runtime handles with startup-resolved services for request contexts.
import type { PluginServicesHandle } from "../plugins/services.js";
import type { createControlUiSessionPullRequestSubscriptions } from "./control-ui-session-pr-subscriptions.js";
import type { HooksConfigResolved } from "./hooks.js";
import { createNativeRoomPresence } from "./native-room-presence.js";
import type { GatewayBroadcastToConnIdsFn } from "./server-broadcast-types.js";
import type { GatewayCronState } from "./server-cron.js";
import {
  createGatewayServerMutableState,
  type GatewayServerMutableState,
} from "./server-runtime-handles.js";
import type { HookClientIpConfig } from "./server/hooks-request-handler.js";
import { createSessionViewerPresenceDeclarations } from "./session-viewer-presence.js";

/** Mutable gateway server state shared across request contexts. */
export type GatewayServerLiveState = GatewayServerMutableState & {
  hooksConfig: HooksConfigResolved | null;
  hookClientIpConfig: HookClientIpConfig;
  cronState: GatewayCronState;
  controlUiSessionPullRequests?: ReturnType<typeof createControlUiSessionPullRequestSubscriptions>;
  sessionViewerPresence?: ReturnType<typeof createSessionViewerPresenceDeclarations>;
  nativeRoomPresence: ReturnType<typeof createNativeRoomPresence>;
  pluginServices: PluginServicesHandle | null;
  gatewayMethods: string[];
};

/** Creates gateway live state with fresh mutable runtime handles. */
export function createGatewayServerLiveState(params: {
  hooksConfig: HooksConfigResolved | null;
  hookClientIpConfig: HookClientIpConfig;
  cronState: GatewayCronState;
  gatewayMethods: string[];
  broadcastToConnIds: GatewayBroadcastToConnIdsFn;
  clients: Parameters<typeof createSessionViewerPresenceDeclarations>[0]["clients"];
  broadcast: Parameters<typeof createSessionViewerPresenceDeclarations>[0]["broadcast"];
  incrementPresenceVersion: () => number;
  getHealthVersion: () => number;
}): GatewayServerLiveState {
  return {
    ...createGatewayServerMutableState(),
    hooksConfig: params.hooksConfig,
    hookClientIpConfig: params.hookClientIpConfig,
    cronState: params.cronState,
    controlUiSessionPullRequests: undefined,
    sessionViewerPresence: createSessionViewerPresenceDeclarations(params),
    nativeRoomPresence: createNativeRoomPresence({
      emit: (event, recipients) =>
        params.broadcastToConnIds("session.presence", event, recipients, {
          dropIfSlow: true,
          sessionKeys: [event.roomKey],
          sessionSubscriptionVerified: true,
        }),
    }),
    pluginServices: null,
    gatewayMethods: params.gatewayMethods,
  };
}
