import type { Static } from "typebox";
import { Type } from "typebox";
import { closedObject } from "./closed-object.js";
import { NonEmptyString } from "./primitives.js";

export const RoomKindSchema = Type.Union([
  Type.Literal("channel"),
  Type.Literal("dm"),
  Type.Literal("group-dm"),
  Type.Literal("thread"),
]);

export const ThreadOriginSchema = closedObject({
  parentRoomKey: NonEmptyString,
  originRootMessageId: Type.Optional(NonEmptyString),
});

export type RoomKind = Static<typeof RoomKindSchema>;
export type ThreadOrigin = Static<typeof ThreadOriginSchema>;
