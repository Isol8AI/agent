import { Value } from "typebox/value";
import { describe, expect, it } from "vitest";
import {
  aggregateNativePresence,
  normalizeNativePresenceEvent,
  reconcileNativePresenceSnapshot,
} from "./native-presence-projection.js";
import {
  SessionsPresenceHeartbeatParamsSchema,
  type NativePresenceEvent,
  type NativePresenceSnapshot,
} from "./schema/sessions-viewer-presence.js";

const at = 1_800_000_000_000;
const event: NativePresenceEvent = {
  roomKey: "agent:main:room",
  actor: { type: "profile", id: "one" },
  connectionId: "connection-one",
  connectionStartedAtMs: at,
  sequence: 1,
  state: "online",
  serverReceivedAtMs: at,
  expiresAtMs: at + 90_000,
  visibility: "visible",
  visibilityObservedAtMs: at,
  recentInput: "recent",
  recentInputObservedAtMs: at,
  viewingIntent: "viewing",
  viewingIntentReceivedAtMs: at,
};
const snapshot: NativePresenceSnapshot = {
  roomKey: event.roomKey,
  serverNowAtMs: at,
  inventoryStatus: "complete",
  connections: [event],
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
        sessionKey: event.roomKey,
        [field]: at,
      }),
    ).toBe(false);
  });

  it.each([at / 1000, at + 1, at + 0.5, "2027-01-15T08:00:00Z", NaN, Infinity])(
    "does not coerce invalid start or observation evidence %s",
    (value) => {
      expect(
        normalizeNativePresenceEvent({ ...event, connectionStartedAtMs: value }, at),
      ).toBeUndefined();
      expect(
        normalizeNativePresenceEvent({ ...event, visibilityObservedAtMs: value }, at),
      ).toMatchObject({ state: "unknown", visibility: "unknown" });
      expect(
        normalizeNativePresenceEvent({ ...event, authoritativeLastSeenAtMs: value }, at)
          ?.authoritativeLastSeenAtMs,
      ).toBeUndefined();
    },
  );

  it("accepts bounded future leases and rejects overlong connection or typing expiry", () => {
    expect(normalizeNativePresenceEvent(event, at)).toEqual(event);
    expect(
      normalizeNativePresenceEvent({ ...event, expiresAtMs: at + 90_001 }, at),
    ).toBeUndefined();
    expect(
      normalizeNativePresenceEvent({ ...event, state: "typing", expiresAtMs: at + 2_500 }, at)
        ?.state,
    ).toBe("typing");
    expect(
      normalizeNativePresenceEvent({ ...event, state: "typing", expiresAtMs: at + 2_501 }, at),
    ).toBeUndefined();
  });

  it("merges connection ids independently, rejects stale and invalid sequence poisoning, and reconciles only complete omissions", () => {
    const initial = reconcileNativePresenceSnapshot(undefined, snapshot);
    const another = { ...event, connectionId: "second", sequence: 1 };
    const partial = reconcileNativePresenceSnapshot(initial, {
      ...snapshot,
      inventoryStatus: "incomplete",
      connections: [another],
    });
    expect(partial.connections).toHaveLength(2);
    const failed = reconcileNativePresenceSnapshot(partial, {
      ...snapshot,
      inventoryStatus: "failed",
      connections: [],
    });
    expect(failed.connections).toHaveLength(2);
    expect(failed.inventoryStatus).toBe("failed");
    const invalid = reconcileNativePresenceSnapshot(failed, {
      ...snapshot,
      connections: [{ ...event, connectionStartedAtMs: at + 1, sequence: 100 }],
    });
    expect(invalid.connections).toHaveLength(2);
    expect(invalid.inventoryStatus).toBe("complete");
    expect(invalid.evidenceStatus).toBe("invalid");
    expect(
      invalid.connections.find((row) => row.connectionId === event.connectionId)?.sequence,
    ).toBe(1);
    const fresh = reconcileNativePresenceSnapshot(invalid, {
      ...snapshot,
      connections: [{ ...event, sequence: 2, state: "away" }],
    });
    expect(fresh.connections).toEqual([expect.objectContaining({ sequence: 2, state: "away" })]);
    const stale = reconcileNativePresenceSnapshot(fresh, snapshot);
    expect(stale.connections[0].sequence).toBe(2);
    expect(aggregateNativePresence(partial, event.actor)).toBe("online");
    expect(
      aggregateNativePresence(
        { ...snapshot, inventoryStatus: "incomplete", connections: [] },
        event.actor,
      ),
    ).toBe("unknown");
    expect(aggregateNativePresence({ ...snapshot, connections: [] }, event.actor)).toBe("offline");
  });
});
