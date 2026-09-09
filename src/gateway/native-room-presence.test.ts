import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createNativeRoomPresence, type NativePresenceAuthority } from "./native-room-presence.js";

const start = 1_800_000_000_000;
const room = "agent:main:room";
const actor = { type: "profile" as const, id: "human-one" };
const authority: NativePresenceAuthority = { actor, isAuthorized: () => true };
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
    expect(presence.snapshot(room).connections).toEqual([]);
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
    expect(presence.snapshot(room).connections[0]).toMatchObject({
      state: "online",
      sequence: 2,
      expiresAtMs: start + 90_000,
    });
    presence.clearViewing("foreground", new Set());
    expect(presence.snapshot(room).connections[0]).toMatchObject({
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
    expect(presence.snapshot(room).connections).toHaveLength(2);
    presence.disconnect("tab-two", true);
    expect(presence.snapshot(room).connections).toEqual([first]);
    const reconnected = presence.update("tab-two-reconnected", room, authority, online)!;
    expect(reconnected.connectionId).not.toBe(second.connectionId);
    expect(reconnected.sequence).toBe(1);
    presence.disconnect("tab-one", true);
    presence.disconnect("tab-two-reconnected", true);
    expect(presence.snapshot(room).connections).toEqual([
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
    expect(presence.snapshot(room).connections[0].state).toBe("typing");
    vi.advanceTimersByTime(1);
    expect(presence.snapshot(room).connections[0]).toMatchObject({
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
    expect(presence.snapshot(room).connections[0].state).toBe("online");
    vi.advanceTimersByTime(26_500);
    for (let heartbeat = 1; heartbeat <= 10; heartbeat += 1) {
      presence.update("one", room, authority, { heartbeat: true });
      if (heartbeat < 10) {
        vi.advanceTimersByTime(30_000);
      }
    }
    expect(presence.snapshot(room).connections[0]).toMatchObject({
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
    expect(presence.snapshot(room).serverNowAtMs).toBe(start + 30_000);
    presence.stop();
  });

  it("revokes every actor lease and its subscription immediately without converting failed ACL reads into offline", () => {
    let access: "allowed" | "denied" | "failed" = "allowed";
    const revocable = {
      actor,
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
    expect(presence.snapshot(room)).toMatchObject({ inventoryStatus: "failed", connections: [] });
    presence.update(
      "other",
      room,
      { actor: { type: "agent", id: "runtime-agent" }, isAuthorized: () => true },
      online,
    );
    expect(presence.snapshot(room)).toMatchObject({
      inventoryStatus: "incomplete",
      connections: [expect.objectContaining({ actor: { type: "agent", id: "runtime-agent" } })],
    });
    access = "denied";
    presence.revalidate();
    expect(
      presence.snapshot(room).connections.filter((event) => event.actor.type === "profile"),
    ).toEqual([expect.objectContaining({ state: "offline", authoritativeLastSeenAtMs: start })]);
    expect(emit.mock.lastCall?.[1]).toEqual(new Set());
    presence.stop();
  });
});
