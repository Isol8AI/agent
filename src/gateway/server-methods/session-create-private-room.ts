import {
  ErrorCodes,
  errorShape,
  type SessionsCreateParams,
} from "../../../packages/gateway-protocol/src/index.js";
import type { RespondFn } from "./types.js";

export function validate(
  params: SessionsCreateParams,
  respond: RespondFn,
): boolean | null {
  const restricted =
    params.visibility === "restricted" ||
    params.roomKind !== undefined ||
    params.members !== undefined ||
    params.threadOrigin !== undefined;
  if (!restricted) {
    return false;
  }
  if (params.visibility !== "restricted" || !params.roomKind || !params.members) {
    respond(
      false,
      undefined,
      errorShape(
        ErrorCodes.INVALID_REQUEST,
        "restricted room creation requires visibility, roomKind, and members",
      ),
    );
    return null;
  }
  if (
    params.incognito === true ||
    params.cwd !== undefined ||
    params.worktree !== undefined ||
    params.worktreeBaseRef !== undefined ||
    params.worktreeName !== undefined ||
    params.projectId !== undefined ||
    params.projectGitUrl !== undefined ||
    params.repository !== undefined ||
    params.execNode !== undefined ||
    params.catalogId !== undefined ||
    params.mentions !== undefined ||
    params.permissionMode !== undefined ||
    params.toolOverrides !== undefined
  ) {
    respond(
      false,
      undefined,
      errorShape(
        ErrorCodes.INVALID_REQUEST,
        "restricted rooms use the server-owned private execution policy",
      ),
    );
    return null;
  }
  return true;
}

export function allowsInitialTurn(
  restricted: boolean,
  hasInitialTurn: boolean,
  respond: RespondFn,
): boolean {
  if (!restricted || !hasInitialTurn) {
    return true;
  }
  respond(
    false,
    undefined,
    errorShape(
      ErrorCodes.INVALID_REQUEST,
      "restricted room creation cannot start a model run; append content separately",
    ),
  );
  return false;
}

export function protocolFields(params: SessionsCreateParams) {
  return {
    visibility: params.visibility,
    roomKind: params.roomKind,
    members: params.members,
    threadOrigin: params.threadOrigin,
  };
}
