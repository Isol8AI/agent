import { randomUUID } from "node:crypto";
import {
  NATIVE_PRESENCE_MAX_CONNECTIONS,
  PRESENCE_AVAILABLE_ACTIVITY_MS,
  PRESENCE_EXPIRES_AFTER_MS,
  TYPING_EXPIRES_AFTER_MS,
  TYPING_THROTTLE_MS,
  type NativePresenceActor,
  type NativePresenceEvent,
  type NativePresenceSnapshot,
} from "../../packages/gateway-protocol/src/schema/sessions-viewer-presence.js";
import { SESSION_VIEWER_PRESENCE_MAX_KEYS } from "../../packages/gateway-protocol/src/schema/sessions-viewer-presence.js";

export type NativePresenceAuthority = {
  actor: NativePresenceActor;
  /** Exact persisted session lifecycle; never accepted from the wire. */
  sessionId: string;
  /** Revalidate the exact room instance and live membership synchronously. */
  isAuthorized: () => boolean;
};
type Lease = {
  authority: NativePresenceAuthority;
  event: NativePresenceEvent;
  leaseExpiresAtMs: number;
  typingAtMs?: number;
  lastTypingAtMs?: number;
};
type Connection = {
  id: string;
  startedAtMs: number;
  sequence: number;
  rooms: Map<string, Lease>;
};

/** Ephemeral leases only: participant history and transcript traffic never enter this owner. */
export function createNativeRoomPresence(params: {
  emit: (event: NativePresenceEvent, recipients: ReadonlySet<string>) => void;
}) {
  // Anchor epoch time to monotonic elapsed time, including across wall-clock corrections.
  const epoch = Date.now();
  const elapsed = performance.now();
  const clock = () => epoch + Math.floor(performance.now() - elapsed);
  const connections = new Map<string, Connection>();
  const subscriptions = new Map<string, Map<string, NativePresenceAuthority>>();
  const tombstones = new Map<string, { sessionId: string; event: NativePresenceEvent }>();
  let lastNow: number | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let stopped = false;
  let leaseCount = 0;
  const now = () => {
    const sampled = clock();
    if (!Number.isSafeInteger(sampled) || sampled < 1_000_000_000_000) {
      throw new Error("native presence server clock unavailable");
    }
    lastNow = Math.max(lastNow ?? sampled, sampled);
    return lastNow;
  };
  const lifecycleKey = (room: string, sessionId: string) => JSON.stringify([room, sessionId]);
  const actorKey = (room: string, sessionId: string, actor: NativePresenceActor) =>
    JSON.stringify([room, sessionId, actor.type, actor.id]);
  const authorized = (authority: NativePresenceAuthority) => authority.isAuthorized();
  const send = (event: NativePresenceEvent, sessionId: string) => {
    const recipients = new Set<string>();
    for (const [connId, rooms] of subscriptions) {
      const authority = rooms.get(event.roomKey);
      if (!authority || authority.sessionId !== sessionId) {
        continue;
      }
      try {
        if (authorized(authority)) {
          recipients.add(connId);
        } else {
          rooms.delete(event.roomKey);
        }
      } catch {
        // Failed ACL evidence cannot authorize delivery or prove revocation.
      }
    }
    params.emit({ ...event, actor: { ...event.actor } }, recipients);
  };
  const publish = (connection: Connection, lease: Lease, at: number) => {
    lease.event.sequence = ++connection.sequence;
    lease.event.serverReceivedAtMs = at;
    send(lease.event, lease.authority.sessionId);
  };
  const project = (lease: Lease, at: number) => {
    const event = lease.event;
    if (
      event.recentInput === "recent" &&
      event.recentInputObservedAtMs !== undefined &&
      at - event.recentInputObservedAtMs >= PRESENCE_AVAILABLE_ACTIVITY_MS
    ) {
      event.recentInput = "stale";
    }
    const typing =
      lease.typingAtMs !== undefined && at < lease.typingAtMs + TYPING_EXPIRES_AFTER_MS;
    event.state = typing
      ? "typing"
      : event.visibility === "hidden" ||
          event.recentInput === "stale" ||
          event.viewingIntent === "not-viewing"
        ? "away"
        : event.visibility === "visible" &&
            event.recentInput === "recent" &&
            event.viewingIntent === "viewing"
          ? "online"
          : "unknown";
    event.expiresAtMs = typing
      ? Math.min(lease.leaseExpiresAtMs, lease.typingAtMs! + TYPING_EXPIRES_AFTER_MS)
      : lease.leaseExpiresAtMs;
  };
  const retire = (connection: Connection, room: string, lease: Lease, at: number) => {
    connection.rooms.delete(room);
    leaseCount -= 1;
    lease.event.state = "offline";
    lease.event.expiresAtMs = at;
    const key = actorKey(room, lease.authority.sessionId, lease.event.actor);
    const remaining = [...connections.values()].some((candidate) => {
      const other = candidate.rooms.get(room);
      return (
        other &&
        other.leaseExpiresAtMs > at &&
        actorKey(room, other.authority.sessionId, other.event.actor) === key
      );
    });
    if (!remaining) {
      lease.event.authoritativeLastSeenAtMs = at;
      tombstones.delete(key);
      tombstones.set(key, { sessionId: lease.authority.sessionId, event: lease.event });
      while (tombstones.size > NATIVE_PRESENCE_MAX_CONNECTIONS) {
        tombstones.delete(tombstones.keys().next().value!);
      }
    }
    publish(connection, lease, at);
  };
  const reconcile = (at: number) => {
    // ponytail: bounded lease scan; index by room if measured fanout requires it.
    const failedLifecycles = new Set<string>();
    for (const [transportId, connection] of connections) {
      for (const [room, lease] of connection.rooms) {
        try {
          if (lease.leaseExpiresAtMs <= at || !authorized(lease.authority)) {
            retire(connection, room, lease, at);
            continue;
          }
          const before = JSON.stringify([lease.event.state, lease.event.recentInput]);
          project(lease, at);
          if (before !== JSON.stringify([lease.event.state, lease.event.recentInput])) {
            publish(connection, lease, at);
          }
        } catch {
          failedLifecycles.add(lifecycleKey(room, lease.authority.sessionId));
        }
      }
      if (connection.rooms.size === 0) {
        connections.delete(transportId);
      }
    }
    for (const [key, { event }] of tombstones) {
      if (at - event.serverReceivedAtMs >= PRESENCE_AVAILABLE_ACTIVITY_MS) {
        tombstones.delete(key);
      }
    }
    for (const [transportId, rooms] of subscriptions) {
      for (const [room, authority] of rooms) {
        try {
          if (!authorized(authority)) {
            rooms.delete(room);
          }
        } catch {
          // Keep the registration on an unavailable read, but send() withholds delivery.
        }
      }
      if (rooms.size === 0) {
        subscriptions.delete(transportId);
      }
    }
    return failedLifecycles;
  };
  const schedule = () => {
    if (timer) {
      clearTimeout(timer);
    }
    timer = undefined;
    if (stopped || connections.size === 0) {
      return;
    }
    const at = now();
    let next = at + PRESENCE_EXPIRES_AFTER_MS;
    for (const connection of connections.values()) {
      for (const lease of connection.rooms.values()) {
        next = Math.min(next, lease.leaseExpiresAtMs);
        if (lease.typingAtMs !== undefined && lease.typingAtMs + TYPING_EXPIRES_AFTER_MS > at) {
          next = Math.min(next, lease.typingAtMs + TYPING_EXPIRES_AFTER_MS);
        }
        if (
          lease.event.recentInput === "recent" &&
          lease.event.recentInputObservedAtMs !== undefined
        ) {
          next = Math.min(
            next,
            lease.event.recentInputObservedAtMs + PRESENCE_AVAILABLE_ACTIVITY_MS,
          );
        }
      }
    }
    timer = setTimeout(
      () => {
        reconcile(now());
        schedule();
      },
      Math.max(1, next - at),
    );
    timer.unref?.();
  };
  const admit = (
    transportId: string,
    room: string,
    authority: NativePresenceAuthority,
    at: number,
  ) => {
    if (stopped || !authorized(authority)) {
      throw new Error("native presence room authorization unavailable");
    }
    let connection = connections.get(transportId);
    if (!connection) {
      if (connections.size >= NATIVE_PRESENCE_MAX_CONNECTIONS) {
        throw new Error("native presence capacity reached");
      }
      connection = { id: randomUUID(), startedAtMs: at, sequence: 0, rooms: new Map() };
      connections.set(transportId, connection);
    }
    let lease = connection.rooms.get(room);
    if (lease && lease.authority.sessionId !== authority.sessionId) {
      retire(connection, room, lease, at);
      lease = undefined;
    }
    if (
      lease &&
      actorKey(room, authority.sessionId, lease.event.actor) !==
        actorKey(room, authority.sessionId, authority.actor)
    ) {
      throw new Error("native presence connection identity changed");
    }
    if (!lease) {
      if (
        connection.rooms.size >= SESSION_VIEWER_PRESENCE_MAX_KEYS ||
        leaseCount >= NATIVE_PRESENCE_MAX_CONNECTIONS
      ) {
        throw new Error("native presence room capacity reached");
      }
      lease = {
        authority,
        leaseExpiresAtMs: at + PRESENCE_EXPIRES_AFTER_MS,
        event: {
          roomKey: room,
          actor: { ...authority.actor },
          connectionId: connection.id,
          connectionStartedAtMs: connection.startedAtMs,
          sequence: 0,
          state: "unknown",
          serverReceivedAtMs: at,
          expiresAtMs: at + PRESENCE_EXPIRES_AFTER_MS,
          visibility: "unknown",
          recentInput: "unknown",
          viewingIntent: "unknown",
        },
      };
      connection.rooms.set(room, lease);
      leaseCount += 1;
      tombstones.delete(actorKey(room, authority.sessionId, authority.actor));
    }
    lease.authority = authority;
    return { connection, lease };
  };
  const update = (
    transportId: string,
    room: string,
    authority: NativePresenceAuthority,
    intent: {
      heartbeat?: boolean;
      visibility?: NativePresenceEvent["visibility"];
      recentInput?: NativePresenceEvent["recentInput"];
      viewingIntent?: NativePresenceEvent["viewingIntent"];
      typing?: boolean;
    },
  ) => {
    if (stopped || !authorized(authority)) {
      throw new Error("native presence room authorization unavailable");
    }
    const at = now();
    reconcile(at);
    const existing = connections.get(transportId)?.rooms.get(room);
    // Typing does not establish or extend a presence lease.
    if (
      intent.typing !== undefined &&
      (!existing || existing.authority.sessionId !== authority.sessionId)
    ) {
      return undefined;
    }
    const { connection, lease } = admit(transportId, room, authority, at);
    if (intent.typing !== undefined) {
      if (lease.lastTypingAtMs !== undefined && at - lease.lastTypingAtMs < TYPING_THROTTLE_MS) {
        return undefined;
      }
      lease.lastTypingAtMs = at;
      lease.typingAtMs = intent.typing ? at : undefined;
    }
    if (intent.heartbeat) {
      lease.leaseExpiresAtMs = at + PRESENCE_EXPIRES_AFTER_MS;
    }
    if (intent.visibility !== undefined) {
      lease.event.visibility = intent.visibility;
      lease.event.visibilityObservedAtMs = at;
    }
    if (intent.recentInput !== undefined) {
      lease.event.recentInput = intent.recentInput;
      lease.event.recentInputObservedAtMs = at;
    }
    if (intent.viewingIntent !== undefined) {
      lease.event.viewingIntent = intent.viewingIntent;
      lease.event.viewingIntentReceivedAtMs = at;
    }
    project(lease, at);
    publish(connection, lease, at);
    schedule();
    return { ...lease.event, actor: { ...lease.event.actor } };
  };
  return {
    update,
    clearViewing(transportId: string, retainedRooms: ReadonlySet<string>) {
      const connection = connections.get(transportId);
      if (!connection) {
        return;
      }
      const at = now();
      reconcile(at);
      for (const [room, lease] of connection.rooms) {
        if (!retainedRooms.has(room) && lease.event.viewingIntent === "viewing") {
          lease.event.viewingIntent = "not-viewing";
          lease.event.viewingIntentReceivedAtMs = at;
          project(lease, at);
          publish(connection, lease, at);
        }
      }
      schedule();
    },
    snapshot(room: string, authority: NativePresenceAuthority): NativePresenceSnapshot {
      const at = now();
      let failed = reconcile(at).has(lifecycleKey(room, authority.sessionId));
      const events: NativePresenceEvent[] = [];
      for (const connection of connections.values()) {
        const lease = connection.rooms.get(room);
        if (!lease || lease.authority.sessionId !== authority.sessionId) {
          continue;
        }
        try {
          if (authorized(lease.authority)) {
            events.push({ ...lease.event, actor: { ...lease.event.actor } });
          }
        } catch {
          failed = true;
        }
      }
      for (const tombstone of tombstones.values()) {
        const { event } = tombstone;
        if (event.roomKey === room && tombstone.sessionId === authority.sessionId) {
          events.push({ ...event, actor: { ...event.actor } });
        }
      }
      failed ||= events.length > NATIVE_PRESENCE_MAX_CONNECTIONS;
      schedule();
      return {
        roomKey: room,
        inventoryStatus: failed ? (events.length ? "incomplete" : "failed") : "complete",
        serverNowAtMs: at,
        connections: events.slice(0, NATIVE_PRESENCE_MAX_CONNECTIONS),
        ...(failed ? { failureCode: "PRESENCE_INVENTORY_INCOMPLETE" } : {}),
      };
    },
    subscribe(transportId: string, room: string, authority: NativePresenceAuthority) {
      if (stopped || !authorized(authority)) {
        throw new Error("native presence subscription denied");
      }
      const rooms = subscriptions.get(transportId) ?? new Map<string, NativePresenceAuthority>();
      if (
        (!subscriptions.has(transportId) &&
          subscriptions.size >= NATIVE_PRESENCE_MAX_CONNECTIONS) ||
        (!rooms.has(room) && rooms.size >= SESSION_VIEWER_PRESENCE_MAX_KEYS)
      ) {
        throw new Error("native presence subscription capacity reached");
      }
      rooms.set(room, authority);
      subscriptions.set(transportId, rooms);
    },
    unsubscribe(transportId: string, room: string) {
      const rooms = subscriptions.get(transportId);
      rooms?.delete(room);
      if (rooms?.size === 0) {
        subscriptions.delete(transportId);
      }
    },
    disconnect(transportId: string, graceful: boolean) {
      subscriptions.delete(transportId);
      const connection = connections.get(transportId);
      if (connection && graceful) {
        const at = now();
        for (const [room, lease] of connection.rooms) {
          retire(connection, room, lease, at);
        }
        connections.delete(transportId);
      }
      schedule();
    },
    revalidate() {
      reconcile(now());
      schedule();
    },
    stop() {
      stopped = true;
      if (timer) {
        clearTimeout(timer);
      }
      connections.clear();
      subscriptions.clear();
      tombstones.clear();
      leaseCount = 0;
    },
  };
}
