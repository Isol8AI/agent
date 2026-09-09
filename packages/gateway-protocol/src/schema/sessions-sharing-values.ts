import type { Static } from "typebox";
import { Type } from "typebox";
import { closedObject } from "./closed-object.js";
import { NonEmptyString } from "./primitives.js";

export const SESSION_VISIBILITY_VALUES = [
  "shared",
  "read-only",
  "suggest",
  "draft",
  "restricted",
] as const;

export const SessionVisibilitySchema = Type.Union([
  Type.Literal("shared"),
  Type.Literal("read-only"),
  Type.Literal("suggest"),
  Type.Literal("draft"),
  Type.Literal("restricted"),
]);

export const SessionSharingRoleSchema = Type.Union([
  Type.Literal("admin"),
  Type.Literal("owner"),
  Type.Literal("member"),
  Type.Literal("viewer"),
]);

/** Stable authorization identity; display actors and participant activity never substitute for it. */
export const SessionMemberIdentitySchema = closedObject({
  type: Type.Union([Type.Literal("profile"), Type.Literal("agent")]),
  id: NonEmptyString,
});

export type SessionVisibility = Static<typeof SessionVisibilitySchema>;
export type SessionSharingRole = Static<typeof SessionSharingRoleSchema>;
export type SessionMemberIdentity = Static<typeof SessionMemberIdentitySchema>;
