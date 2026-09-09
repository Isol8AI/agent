import { Value } from "typebox/value";
import {
  NativePresenceEventSchema,
  NativePresenceSnapshotSchema,
  NATIVE_PRESENCE_MAX_CONNECTIONS,
  PRESENCE_AVAILABLE_ACTIVITY_MS,
  PRESENCE_EXPIRES_AFTER_MS,
  TYPING_EXPIRES_AFTER_MS,
  type NativePresenceEvent,
  type NativePresenceSnapshot,
} from "./schema/sessions-viewer-presence.js";

function epochMs(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 1_000_000_000_000;
}
function observed(value: unknown, serverNowAtMs: number): value is number {
  return epochMs(value) && value <= serverNowAtMs;
}

/** Validate evidence before ordering: invalid large sequences cannot poison the watermark. */
export function normalizeNativePresenceEvent(
  input: unknown,
  serverNowAtMs: number,
): NativePresenceEvent | undefined {
  if (!epochMs(serverNowAtMs) || !input || typeof input !== "object" || Array.isArray(input)) {
    return undefined;
  }
  const event: Record<string, unknown> = { ...input };
  if (
    !observed(event.connectionStartedAtMs, serverNowAtMs) ||
    !observed(event.serverReceivedAtMs, serverNowAtMs) ||
    event.connectionStartedAtMs > event.serverReceivedAtMs ||
    !epochMs(event.expiresAtMs)
  ) {
    return undefined;
  }
  const duration = event.state === "typing" ? TYPING_EXPIRES_AFTER_MS : PRESENCE_EXPIRES_AFTER_MS;
  if (
    event.expiresAtMs < event.serverReceivedAtMs ||
    event.expiresAtMs > event.serverReceivedAtMs + duration
  ) {
    return undefined;
  }
  for (const [field, projection] of [
    ["visibilityObservedAtMs", "visibility"],
    ["recentInputObservedAtMs", "recentInput"],
    ["viewingIntentReceivedAtMs", "viewingIntent"],
  ] as const) {
    if (!observed(event[field], serverNowAtMs) || event[field] > event.serverReceivedAtMs) {
      delete event[field];
      event[projection] = "unknown";
    }
  }
  if (
    !observed(event.authoritativeLastSeenAtMs, serverNowAtMs) ||
    event.authoritativeLastSeenAtMs > event.serverReceivedAtMs
  ) {
    delete event.authoritativeLastSeenAtMs;
  }
  if (!Value.Check(NativePresenceEventSchema, event)) {
    return undefined;
  }
  if (
    event.state === "online" &&
    (event.visibility !== "visible" ||
      event.recentInput !== "recent" ||
      event.viewingIntent !== "viewing")
  ) {
    event.state = "unknown";
  }
  return event;
}

export type NativePresenceInventory = NativePresenceSnapshot & {
  /** Local validation never rewrites the native producer's inventoryStatus. */
  evidenceStatus?: "valid" | "invalid";
  validationFailureCode?: string;
};

/** Merge only validated connection-local evidence; partial/failed omissions are never deletions. */
export function reconcileNativePresenceSnapshot(
  previous: NativePresenceInventory | undefined,
  snapshot: NativePresenceSnapshot,
): NativePresenceInventory {
  if (
    !Array.isArray(snapshot.connections) ||
    snapshot.connections.length > NATIVE_PRESENCE_MAX_CONNECTIONS ||
    !Value.Check(NativePresenceSnapshotSchema, { ...snapshot, connections: [] }) ||
    (previous && snapshot.roomKey !== previous.roomKey)
  ) {
    throw new Error("invalid native presence snapshot boundary");
  }
  if (previous && snapshot.serverNowAtMs < previous.serverNowAtMs) {
    return previous;
  }
  if (snapshot.inventoryStatus === "failed") {
    return {
      ...snapshot,
      connections: previous?.connections ?? [],
      evidenceStatus: previous?.evidenceStatus,
      validationFailureCode: previous?.validationFailureCode,
    };
  }
  const retained = new Map(
    (previous?.connections ?? []).map((event) => [event.connectionId, event]),
  );
  const included = new Set<string>();
  let invalid = false;
  for (const input of snapshot.connections) {
    const event = normalizeNativePresenceEvent(input, snapshot.serverNowAtMs);
    const invalidObservation =
      event &&
      (
        [
          "visibilityObservedAtMs",
          "recentInputObservedAtMs",
          "viewingIntentReceivedAtMs",
          "authoritativeLastSeenAtMs",
        ] as const
      ).some((field) => input[field] !== undefined && event[field] === undefined);
    if (!event || event.roomKey !== snapshot.roomKey || invalidObservation) {
      invalid = true;
      const prior = retained.get(input?.connectionId);
      if (prior) {
        retained.set(prior.connectionId, {
          ...prior,
          state: "unknown",
          visibility: "unknown",
          visibilityObservedAtMs: undefined,
          recentInput: "unknown",
          recentInputObservedAtMs: undefined,
          viewingIntent: "unknown",
          viewingIntentReceivedAtMs: undefined,
          authoritativeLastSeenAtMs: undefined,
        });
      }
      continue;
    }
    included.add(event.connectionId);
    const prior = retained.get(event.connectionId);
    if (
      prior &&
      (event.connectionStartedAtMs !== prior.connectionStartedAtMs ||
        event.actor.type !== prior.actor.type ||
        event.actor.id !== prior.actor.id)
    ) {
      invalid = true;
      retained.set(event.connectionId, { ...prior, state: "unknown" });
      continue;
    }
    if (
      prior &&
      (event.sequence <= prior.sequence || event.serverReceivedAtMs < prior.serverReceivedAtMs)
    ) {
      continue;
    }
    retained.set(event.connectionId, event);
  }
  // An invalid row cannot turn its missing connection into complete negative evidence.
  if (snapshot.inventoryStatus === "complete" && !invalid) {
    for (const [id, event] of retained) {
      if (!included.has(id) && event.serverReceivedAtMs <= snapshot.serverNowAtMs) {
        retained.delete(id);
      }
    }
  }
  if (retained.size > NATIVE_PRESENCE_MAX_CONNECTIONS) {
    return {
      ...snapshot,
      evidenceStatus: "invalid",
      validationFailureCode: "PRESENCE_CAPACITY_EXCEEDED",
      connections: previous?.connections ?? [],
    };
  }
  return {
    ...snapshot,
    connections: [...retained.values()],
    evidenceStatus: invalid ? "invalid" : "valid",
    ...(invalid ? { validationFailureCode: "PRESENCE_INVALID_EVIDENCE" } : {}),
  };
}

/** Supply server time advanced by monotonic elapsed time, never browser wall-clock time. */
export function aggregateNativePresence(
  inventory: NativePresenceInventory,
  actor: NativePresenceEvent["actor"],
  monotonicElapsedMs = 0,
): NativePresenceEvent["state"] {
  if (!Number.isFinite(monotonicElapsedMs) || monotonicElapsedMs < 0) {
    return "unknown";
  }
  const at = inventory.serverNowAtMs + Math.floor(monotonicElapsedMs);
  const events = inventory.connections.filter(
    (event) => event.actor.type === actor.type && event.actor.id === actor.id,
  );
  const live = events.filter((event) => event.state !== "offline" && event.expiresAtMs > at);
  if (live.some((event) => event.state === "typing")) {
    return "typing";
  }
  if (
    live.some(
      (event) =>
        event.state === "online" &&
        event.recentInputObservedAtMs !== undefined &&
        at - event.recentInputObservedAtMs < PRESENCE_AVAILABLE_ACTIVITY_MS,
    )
  ) {
    return "online";
  }
  if (live.some((event) => event.state === "away")) {
    return "away";
  }
  if (live.length) {
    return "unknown";
  }
  // Expiry alone cannot invent server-owned last seen; typing expires independently of its lease.
  return inventory.inventoryStatus === "complete" &&
    inventory.evidenceStatus !== "invalid" &&
    events.every(
      (event) => event.state === "offline" || (event.state !== "typing" && event.expiresAtMs <= at),
    )
    ? "offline"
    : "unknown";
}
