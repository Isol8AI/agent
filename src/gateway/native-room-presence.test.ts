import { expectDefined } from "@openclaw/normalization-core";
import { Value } from "typebox/value";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  aggregateNativePresence,
  normalizeNativePresenceEvent,
  reconcileNativePresenceSnapshot,
} from "../../packages/gateway-protocol/src/native-presence-projection.js";
import {
  SessionsPresenceHeartbeatParamsSchema,
  type NativePresenceEvent,
  type NativePresenceSnapshot,
} from "../../packages/gateway-protocol/src/schema/sessions-viewer-presence.js";
import { createNativeRoomPresence, type NativePresenceAuthority } from "./native-room-presence.js";
import { nativePresenceHandlers } from "./server-methods/sessions-presence.js";
import type {
  GatewayRequestContext,
  GatewayRequestHandlerOptions,
} from "./server-methods/types.js";

const access = vi.hoisted(() => ({ allowed: true }));
vi.mock("./native-room-presence-authority.js", () => ({
  prepareNativeRoomPresenceAuthority: ({ sessionKey }: { sessionKey: string }) =>
    access.allowed
      ? {
          roomKey: sessionKey,
          authority: {
            actor: { type: "profile", id: "server-profile" },
            sessionId: "session-room",
            isAuthorized: () => access.allowed,
          },
        }
      : undefined,
}));

const start = 1_800_000_000_000;
const room = "agent:main:room";
const actor = { type: "profile" as const, id: "human-one" };
const authority: NativePresenceAuthority = {
  actor,
  sessionId: "session-room",
  isAuthorized: () => true,
};
const online = {
  heartbeat: true,
  visibility: "visible" as const,
  recentInput: "recent" as const,
  viewingIntent: "viewing" as const,
};

describe("native room presence lifecycle", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date", "performance", "setTimeout", "clearTimeout"] });
    vi.setSystemTime(start);
  });
  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it("keeps subscription and viewing independent and stamps all evidence on the server clock", () => {
    const emit = vi.fn();
    const presence = createNativeRoomPresence({ emit });
    presence.subscribe("background", room, authority);
    expect(presence.snapshot(room, authority).connections).toEqual([]);
    const viewing = presence.update("foreground", room, authority, { viewingIntent: "viewing" });
    expect(viewing).toMatchObject({
      state: "unknown",
      sequence: 1,
      visibility: "unknown",
      recentInput: "unknown",
      viewingIntentReceivedAtMs: start,
    });
    expect(emit.mock.lastCall?.[1]).toEqual(new Set(["background"]));
    presence.update("foreground", room, authority, {
      heartbeat: true,
      visibility: "visible",
      recentInput: "recent",
    });
    expect(presence.snapshot(room, authority).connections[0]).toMatchObject({
      state: "online",
      sequence: 2,
      expiresAtMs: start + 90_000,
    });
    presence.clearViewing("foreground", new Set());
    expect(presence.snapshot(room, authority).connections[0]).toMatchObject({
      state: "away",
      viewingIntent: "not-viewing",
    });
    expect(emit.mock.lastCall?.[1]).toEqual(new Set(["background"]));
    presence.stop();
  });

  it("aggregates simultaneous tabs, creates a new reconnect id, and authors last seen only at the final disconnect", () => {
    const presence = createNativeRoomPresence({ emit: vi.fn() });
    const first = presence.update("tab-one", room, authority, online)!;
    vi.advanceTimersByTime(1);
    const second = presence.update("tab-two", room, authority, online)!;
    expect(second.connectionId).not.toBe(first.connectionId);
    expect(presence.snapshot(room, authority).connections).toHaveLength(2);
    presence.disconnect("tab-two", true);
    expect(presence.snapshot(room, authority).connections).toEqual([first]);
    const reconnected = presence.update("tab-two-reconnected", room, authority, online)!;
    expect(reconnected.connectionId).not.toBe(second.connectionId);
    expect(reconnected.sequence).toBe(1);
    presence.disconnect("tab-one", true);
    presence.disconnect("tab-two-reconnected", true);
    expect(presence.snapshot(room, authority).connections).toEqual([
      expect.objectContaining({ state: "offline", authoritativeLastSeenAtMs: start + 1 }),
    ]);
    presence.stop();
  });

  it("expires abnormal disconnects at 90 seconds and does not let typing renew a lease", () => {
    const presence = createNativeRoomPresence({ emit: vi.fn() });
    presence.update("one", room, authority, online);
    vi.advanceTimersByTime(89_000);
    expect(presence.update("one", room, authority, { typing: true })).toMatchObject({
      state: "typing",
      expiresAtMs: start + 90_000,
    });
    presence.disconnect("one", false);
    vi.advanceTimersByTime(999);
    expect(presence.snapshot(room, authority).connections[0].state).toBe("typing");
    vi.advanceTimersByTime(1);
    expect(presence.snapshot(room, authority).connections[0]).toMatchObject({
      state: "offline",
      authoritativeLastSeenAtMs: start + 90_000,
    });
    presence.stop();
  });

  it("throttles typing to one second, expires it at 2.5 seconds, and ages activity after five minutes of heartbeats", () => {
    const presence = createNativeRoomPresence({ emit: vi.fn() });
    presence.update("one", room, authority, online);
    presence.update("one", room, authority, { typing: true });
    vi.advanceTimersByTime(999);
    expect(presence.update("one", room, authority, { typing: true })).toBeUndefined();
    vi.advanceTimersByTime(1);
    expect(presence.update("one", room, authority, { typing: true })).toMatchObject({
      state: "typing",
      expiresAtMs: start + 3_500,
    });
    vi.advanceTimersByTime(2_500);
    expect(presence.snapshot(room, authority).connections[0].state).toBe("online");
    vi.advanceTimersByTime(26_500);
    for (let heartbeat = 1; heartbeat <= 10; heartbeat += 1) {
      presence.update("one", room, authority, { heartbeat: true });
      if (heartbeat < 10) {
        vi.advanceTimersByTime(30_000);
      }
    }
    expect(presence.snapshot(room, authority).connections[0]).toMatchObject({
      state: "away",
      recentInput: "stale",
      recentInputObservedAtMs: start,
    });
    presence.stop();
  });

  it("keeps server time monotonic when the host wall clock moves backward", () => {
    const presence = createNativeRoomPresence({ emit: vi.fn() });
    presence.update("one", room, authority, online);
    vi.advanceTimersByTime(30_000);
    vi.setSystemTime(start - 500_000);
    const event = presence.update("one", room, authority, { heartbeat: true });
    expect(event).toMatchObject({ serverReceivedAtMs: start + 30_000, sequence: 2 });
    expect(presence.snapshot(room, authority).serverNowAtMs).toBe(start + 30_000);
    presence.stop();
  });

  it("revokes every actor lease and its subscription immediately without converting failed ACL reads into offline", () => {
    let access: "allowed" | "denied" | "failed" = "allowed";
    const revocable = {
      actor,
      sessionId: "session-room",
      isAuthorized: () => {
        if (access === "failed") {
          throw new Error("storage unavailable");
        }
        return access === "allowed";
      },
    };
    const emit = vi.fn();
    const presence = createNativeRoomPresence({ emit });
    presence.subscribe("one", room, revocable);
    presence.update("one", room, revocable, online);
    presence.update("two", room, revocable, online);
    access = "failed";
    expect(presence.snapshot(room, revocable)).toMatchObject({
      inventoryStatus: "failed",
      connections: [],
    });
    presence.update(
      "other",
      room,
      {
        actor: { type: "agent", id: "runtime-agent" },
        sessionId: "session-room",
        isAuthorized: () => true,
      },
      online,
    );
    expect(presence.snapshot(room, revocable)).toMatchObject({
      inventoryStatus: "incomplete",
      connections: [expect.objectContaining({ actor: { type: "agent", id: "runtime-agent" } })],
    });
    access = "denied";
    presence.revalidate();
    expect(
      presence
        .snapshot(room, { ...revocable, isAuthorized: () => true })
        .connections.filter((event) => event.actor.type === "profile"),
    ).toEqual([expect.objectContaining({ state: "offline", authoritativeLastSeenAtMs: start })]);
    expect(emit.mock.lastCall?.[1]).toEqual(new Set());
    presence.stop();
  });

  it("isolates active leases, subscriptions, and tombstones by exact session lifecycle", () => {
    let currentSessionId = "session-one";
    const firstAuthority: NativePresenceAuthority = {
      actor,
      sessionId: "session-one",
      isAuthorized: () => currentSessionId === "session-one",
    };
    const secondAuthority: NativePresenceAuthority = {
      actor,
      sessionId: "session-two",
      isAuthorized: () => currentSessionId === "session-two",
    };
    const thirdAuthority: NativePresenceAuthority = {
      actor,
      sessionId: "session-three",
      isAuthorized: () => currentSessionId === "session-three",
    };
    const emit = vi.fn();
    const presence = createNativeRoomPresence({ emit });
    presence.subscribe("first-observer", room, firstAuthority);
    presence.update("actor-tab", room, firstAuthority, online);

    currentSessionId = "session-two";
    presence.subscribe("second-observer", room, secondAuthority);
    expect(presence.snapshot(room, secondAuthority)).toMatchObject({
      inventoryStatus: "complete",
      connections: [],
    });
    expect(emit.mock.lastCall?.[1]).toEqual(new Set());
    presence.update("actor-tab", room, secondAuthority, online);
    expect(presence.snapshot(room, secondAuthority).connections).toEqual([
      expect.objectContaining({ state: "online" }),
    ]);
    presence.disconnect("actor-tab", true);
    expect(presence.snapshot(room, secondAuthority).connections).toEqual([
      expect.objectContaining({ state: "offline" }),
    ]);

    currentSessionId = "session-three";
    expect(presence.snapshot(room, thirdAuthority)).toMatchObject({
      inventoryStatus: "complete",
      connections: [],
    });
    presence.stop();
  });
});

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
    expect(
      presence.snapshot(room, {
        actor: { type: "profile", id: "server-profile" },
        sessionId: "session-room",
        isAuthorized: () => true,
      }).connections,
    ).toEqual([]);
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

const projectionAt = 1_800_000_000_000;
const projectionEvent: NativePresenceEvent = {
  roomKey: room,
  actor: { type: "profile", id: "one" },
  connectionId: "connection-one",
  connectionStartedAtMs: projectionAt,
  sequence: 1,
  state: "online",
  serverReceivedAtMs: projectionAt,
  expiresAtMs: projectionAt + 90_000,
  visibility: "visible",
  visibilityObservedAtMs: projectionAt,
  recentInput: "recent",
  recentInputObservedAtMs: projectionAt,
  viewingIntent: "viewing",
  viewingIntentReceivedAtMs: projectionAt,
};
const projectionSnapshot: NativePresenceSnapshot = {
  roomKey: projectionEvent.roomKey,
  serverNowAtMs: projectionAt,
  inventoryStatus: "complete",
  connections: [projectionEvent],
};

describe("native presence evidence", () => {
  it.each([
    "actor",
    "sequence",
    "connectionId",
    "connectionStartedAtMs",
    "serverReceivedAtMs",
    "expiresAtMs",
    "lastSeen",
    "lastSeenAt",
    "authoritativeLastSeenAtMs",
    "visibilityObservedAtMs",
    "recentInputObservedAtMs",
  ])("rejects client-authored %s", (field) => {
    expect(
      Value.Check(SessionsPresenceHeartbeatParamsSchema, {
        sessionKey: projectionEvent.roomKey,
        [field]: projectionAt,
      }),
    ).toBe(false);
  });

  it.each([
    projectionAt / 1000,
    projectionAt + 1,
    projectionAt + 0.5,
    "2027-01-15T08:00:00Z",
    NaN,
    Infinity,
  ])("does not coerce invalid start or observation evidence %s", (value) => {
    expect(
      normalizeNativePresenceEvent(
        { ...projectionEvent, connectionStartedAtMs: value },
        projectionAt,
      ),
    ).toBeUndefined();
    expect(
      normalizeNativePresenceEvent(
        { ...projectionEvent, visibilityObservedAtMs: value },
        projectionAt,
      ),
    ).toMatchObject({ state: "unknown", visibility: "unknown" });
    expect(
      normalizeNativePresenceEvent(
        { ...projectionEvent, authoritativeLastSeenAtMs: value },
        projectionAt,
      )?.authoritativeLastSeenAtMs,
    ).toBeUndefined();
  });

  it("accepts bounded future leases and rejects overlong connection or typing expiry", () => {
    expect(normalizeNativePresenceEvent(projectionEvent, projectionAt)).toEqual(projectionEvent);
    expect(
      normalizeNativePresenceEvent(
        { ...projectionEvent, expiresAtMs: projectionAt + 90_001 },
        projectionAt,
      ),
    ).toBeUndefined();
    expect(
      normalizeNativePresenceEvent(
        { ...projectionEvent, state: "typing", expiresAtMs: projectionAt + 2_500 },
        projectionAt,
      )?.state,
    ).toBe("typing");
    expect(
      normalizeNativePresenceEvent(
        { ...projectionEvent, state: "typing", expiresAtMs: projectionAt + 2_501 },
        projectionAt,
      ),
    ).toBeUndefined();
  });

  it("merges connection ids independently, rejects stale and invalid sequence poisoning, and reconciles only complete omissions", () => {
    const initial = reconcileNativePresenceSnapshot(undefined, projectionSnapshot);
    const another = { ...projectionEvent, connectionId: "second", sequence: 1 };
    const partial = reconcileNativePresenceSnapshot(initial, {
      ...projectionSnapshot,
      inventoryStatus: "incomplete",
      connections: [another],
    });
    expect(partial.connections).toHaveLength(2);
    const failed = reconcileNativePresenceSnapshot(partial, {
      ...projectionSnapshot,
      inventoryStatus: "failed",
      connections: [],
    });
    expect(failed.connections).toHaveLength(2);
    expect(failed.inventoryStatus).toBe("failed");
    const invalid = reconcileNativePresenceSnapshot(failed, {
      ...projectionSnapshot,
      connections: [{ ...projectionEvent, connectionStartedAtMs: projectionAt + 1, sequence: 100 }],
    });
    expect(invalid.connections).toHaveLength(2);
    expect(invalid.inventoryStatus).toBe("complete");
    expect(invalid.evidenceStatus).toBe("invalid");
    expect(
      invalid.connections.find((row) => row.connectionId === projectionEvent.connectionId)
        ?.sequence,
    ).toBe(1);
    const fresh = reconcileNativePresenceSnapshot(invalid, {
      ...projectionSnapshot,
      connections: [{ ...projectionEvent, sequence: 2, state: "away" }],
    });
    expect(fresh.connections).toEqual([expect.objectContaining({ sequence: 2, state: "away" })]);
    const stale = reconcileNativePresenceSnapshot(fresh, projectionSnapshot);
    expect(stale.connections[0].sequence).toBe(2);
    expect(aggregateNativePresence(partial, projectionEvent.actor)).toBe("online");
    expect(
      aggregateNativePresence(
        { ...projectionSnapshot, inventoryStatus: "incomplete", connections: [] },
        projectionEvent.actor,
      ),
    ).toBe("unknown");
    expect(
      aggregateNativePresence({ ...projectionSnapshot, connections: [] }, projectionEvent.actor),
    ).toBe("offline");
  });
});
