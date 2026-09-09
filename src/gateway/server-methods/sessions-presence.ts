import {
  ErrorCodes,
  errorShape,
  validateSessionsPresenceHeartbeatParams as validateHeartbeat,
  validateSessionsPresenceParams as validateTarget,
} from "../../../packages/gateway-protocol/src/index.js";
import { prepareNativeRoomPresenceAuthority } from "../native-room-presence-authority.js";
import type { GatewayRequestHandler, GatewayRequestHandlers } from "./types.js";
import { assertValidParams } from "./validation.js";

function presenceHandler(
  action: "heartbeat" | "snapshot" | "subscribe" | "unsubscribe",
): GatewayRequestHandler {
  return ({ params, client, context, respond }) => {
    const method = `sessions.presence.${action}`;
    if (
      !assertValidParams(
        params,
        action === "heartbeat" ? validateHeartbeat : validateTarget,
        method,
        respond,
      )
    ) {
      return;
    }
    const presence = context.nativeRoomPresence;
    if (!presence || !client?.connId || !context.isConnectionActive?.(client.connId)) {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.UNAVAILABLE, "native room presence unavailable"),
      );
      return;
    }
    try {
      const prepared = prepareNativeRoomPresenceAuthority({
        client,
        getConfig: context.getRuntimeConfig,
        sessionKey: params.sessionKey,
        agentId: params.agentId,
      });
      if (!prepared) {
        respond(
          false,
          undefined,
          errorShape(
            ErrorCodes.INVALID_REQUEST,
            "native room presence requires current room access and an authenticated profile",
          ),
        );
        return;
      }
      const { roomKey, authority } = prepared;
      if (action === "subscribe") {
        presence.subscribe(client.connId, roomKey, authority);
      }
      if (action === "unsubscribe") {
        presence.unsubscribe(client.connId, roomKey);
      }
      if (action === "heartbeat" && validateHeartbeat(params)) {
        presence.update(client.connId, roomKey, authority, {
          heartbeat: true,
          visibility: params.visibility,
          recentInput: params.recentInput,
        });
      }
      respond(true, presence.snapshot(roomKey, authority));
    } catch {
      // Authorization/producer failure never becomes an authoritative empty snapshot.
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.UNAVAILABLE, "native room presence authority unavailable"),
      );
    }
  };
}

export const nativePresenceHandlers: GatewayRequestHandlers = {
  "sessions.presence.heartbeat": presenceHandler("heartbeat"),
  "sessions.presence.snapshot": presenceHandler("snapshot"),
  "sessions.presence.subscribe": presenceHandler("subscribe"),
  "sessions.presence.unsubscribe": presenceHandler("unsubscribe"),
};
