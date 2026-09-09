import type { Static } from "typebox";
import { Type } from "typebox";
import { closedObject } from "./closed-object.js";
import { ChatSendSessionKeyString, NonEmptyString } from "./primitives.js";

/** Maximum sessions one connection may declare as concurrently visible. */
export const SESSION_VIEWER_PRESENCE_MAX_KEYS = 32;

/** Replaces the sessions this connection is currently rendering. */
export const SessionsViewerPresenceSetParamsSchema = closedObject({
  agentId: Type.Optional(NonEmptyString),
  sessionKeys: Type.Array(ChatSendSessionKeyString, {
    maxItems: SESSION_VIEWER_PRESENCE_MAX_KEYS,
  }),
});

/** Canonical session keys retained for this connection's viewer presence. */
export const SessionsViewerPresenceSetResultSchema = closedObject({
  sessionKeys: Type.Array(ChatSendSessionKeyString, {
    maxItems: SESSION_VIEWER_PRESENCE_MAX_KEYS,
  }),
});

export type SessionsViewerPresenceSetParams = Static<typeof SessionsViewerPresenceSetParamsSchema>;
export type SessionsViewerPresenceSetResult = Static<typeof SessionsViewerPresenceSetResultSchema>;

export const NATIVE_PRESENCE_CONTRACT_VERSION = 1;
export const PRESENCE_HEARTBEAT_INTERVAL_MS = 30_000;
export const PRESENCE_EXPIRES_AFTER_MS = 90_000;
export const PRESENCE_AVAILABLE_ACTIVITY_MS = 300_000;
export const TYPING_THROTTLE_MS = 1_000;
export const TYPING_EXPIRES_AFTER_MS = 2_500;
export const NATIVE_PRESENCE_MAX_CONNECTIONS = 4_096;

const EpochMs = Type.Integer({
  minimum: 1_000_000_000_000,
  maximum: Number.MAX_SAFE_INTEGER,
});
const Visibility = Type.Union([
  Type.Literal("visible"),
  Type.Literal("hidden"),
  Type.Literal("unknown"),
]);
const RecentInput = Type.Union([
  Type.Literal("recent"),
  Type.Literal("stale"),
  Type.Literal("unknown"),
]);
const ViewingIntent = Type.Union([
  Type.Literal("viewing"),
  Type.Literal("not-viewing"),
  Type.Literal("unknown"),
]);
export const NativePresenceActorSchema = closedObject({
  type: Type.Union([Type.Literal("profile"), Type.Literal("agent")]),
  id: Type.String({ minLength: 1, maxLength: 256 }),
});
export const NativePresenceEventSchema = closedObject({
  roomKey: ChatSendSessionKeyString,
  actor: NativePresenceActorSchema,
  connectionId: Type.String({ minLength: 1, maxLength: 256 }),
  connectionStartedAtMs: EpochMs,
  sequence: Type.Integer({ minimum: 1, maximum: Number.MAX_SAFE_INTEGER }),
  state: Type.Union([
    Type.Literal("online"),
    Type.Literal("away"),
    Type.Literal("offline"),
    Type.Literal("typing"),
    Type.Literal("unknown"),
  ]),
  serverReceivedAtMs: EpochMs,
  expiresAtMs: EpochMs,
  visibility: Visibility,
  visibilityObservedAtMs: Type.Optional(EpochMs),
  recentInput: RecentInput,
  recentInputObservedAtMs: Type.Optional(EpochMs),
  viewingIntent: ViewingIntent,
  viewingIntentReceivedAtMs: Type.Optional(EpochMs),
  authoritativeLastSeenAtMs: Type.Optional(EpochMs),
});
export const NativePresenceSnapshotSchema = closedObject({
  roomKey: ChatSendSessionKeyString,
  inventoryStatus: Type.Union([
    Type.Literal("complete"),
    Type.Literal("incomplete"),
    Type.Literal("failed"),
  ]),
  serverNowAtMs: EpochMs,
  connections: Type.Array(NativePresenceEventSchema, { maxItems: NATIVE_PRESENCE_MAX_CONNECTIONS }),
  failureCode: Type.Optional(Type.String({ minLength: 1, maxLength: 128 })),
});
export const SessionsPresenceParamsSchema = closedObject({
  sessionKey: ChatSendSessionKeyString,
  agentId: Type.Optional(NonEmptyString),
});
export const SessionsPresenceHeartbeatParamsSchema = closedObject({
  sessionKey: ChatSendSessionKeyString,
  agentId: Type.Optional(NonEmptyString),
  visibility: Type.Optional(Visibility),
  recentInput: Type.Optional(RecentInput),
});

export type NativePresenceActor = Static<typeof NativePresenceActorSchema>;
export type NativePresenceEvent = Static<typeof NativePresenceEventSchema>;
export type NativePresenceSnapshot = Static<typeof NativePresenceSnapshotSchema>;
export type NativeInventoryStatus = NativePresenceSnapshot["inventoryStatus"];
export type NativePresenceState = NativePresenceEvent["state"];
export type NativeVisibilityProjection = NativePresenceEvent["visibility"];
export type NativeRecentInputProjection = NativePresenceEvent["recentInput"];
export type NativeViewingIntent = NativePresenceEvent["viewingIntent"];
export type SessionsPresenceHeartbeatParams = Static<typeof SessionsPresenceHeartbeatParamsSchema>;
