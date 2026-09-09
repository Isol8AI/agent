import { Type, type Static } from "typebox";
import { closedObject } from "./closed-object.js";
import { ChatSendSessionKeyString, NonEmptyString } from "./primitives.js";

export const SessionExecutionDispatchParamsSchema = closedObject({
  sessionKey: ChatSendSessionKeyString,
  expectedSessionId: NonEmptyString,
  inputMessageId: NonEmptyString,
  idempotencyKey: NonEmptyString,
  hopCount: Type.Optional(Type.Integer({ minimum: 0, maximum: 3 })),
});
export type SessionExecutionDispatchParams = Static<typeof SessionExecutionDispatchParamsSchema>;
