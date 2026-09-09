import { expectDefined } from "@openclaw/normalization-core";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createNativeRoomPresence } from "../native-room-presence.js";
import { nativePresenceHandlers } from "./sessions-presence.js";
import type { GatewayRequestContext, GatewayRequestHandlerOptions } from "./types.js";

const access = vi.hoisted(() => ({ allowed: true }));
vi.mock("../native-room-presence-authority.js", () => ({
  prepareNativeRoomPresenceAuthority: ({ sessionKey }: { sessionKey: string }) =>
    access.allowed
      ? {
          roomKey: sessionKey,
          authority: {
            actor: { type: "profile", id: "server-profile" },
            isAuthorized: () => access.allowed,
          },
        }
      : undefined,
}));

describe("native presence Gateway requests", () => {
  afterEach(() => {
    access.allowed = true;
  });

  it("admits subscription without viewing or a lease and rejects forged timestamps before mutation", async () => {
    const presence = createNativeRoomPresence({ emit: vi.fn() });
    const context = {
      nativeRoomPresence: presence,
      isConnectionActive: () => true,
      getRuntimeConfig: () => ({}),
    } as unknown as GatewayRequestContext;
    const call = async (action: string, params: Record<string, unknown>) => {
      const respond = vi.fn();
      const method = `sessions.presence.${action}`;
      await expectDefined(
        nativePresenceHandlers[method],
        method,
      )({
        req: { id: "presence-test", method, type: "req" },
        params,
        respond,
        context,
        client: { connId: "socket-one" } as never,
        isWebchatConnect: () => false,
      } satisfies GatewayRequestHandlerOptions);
      return respond;
    };
    const room = "agent:main:room";
    const subscribed = await call("subscribe", { sessionKey: room });
    expect(subscribed).toHaveBeenCalledWith(
      true,
      expect.objectContaining({ inventoryStatus: "complete", connections: [] }),
    );
    const forged = await call("heartbeat", {
      sessionKey: room,
      authoritativeLastSeenAtMs: Date.now(),
    });
    expect(forged).toHaveBeenCalledWith(
      false,
      undefined,
      expect.objectContaining({ code: "INVALID_REQUEST" }),
    );
    expect(presence.snapshot(room).connections).toEqual([]);
    const heartbeat = await call("heartbeat", {
      sessionKey: room,
      visibility: "visible",
      recentInput: "recent",
    });
    expect(heartbeat).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        connections: [
          expect.objectContaining({
            actor: { type: "profile", id: "server-profile" },
            state: "unknown",
            viewingIntent: "unknown",
          }),
        ],
      }),
    );
    access.allowed = false;
    const revoked = await call("snapshot", { sessionKey: room });
    expect(revoked).toHaveBeenCalledWith(
      false,
      undefined,
      expect.objectContaining({ code: "INVALID_REQUEST" }),
    );
    presence.stop();
  });
});
