import {
  ErrorCodes,
  errorShape,
  type ErrorShape,
  type SessionsCreateParams,
} from "../../../packages/gateway-protocol/src/index.js";

export function validatePrivateRoomCreation(params: SessionsCreateParams): {
  restricted: boolean;
  error: ErrorShape | null;
} {
  const restricted =
    params.visibility === "restricted" ||
    params.roomKind !== undefined ||
    params.members !== undefined ||
    params.threadOrigin !== undefined;
  if (!restricted) {
    return { restricted, error: null };
  }
  if (params.visibility !== "restricted" || !params.roomKind || !params.members) {
    return {
      restricted,
      error: errorShape(
        ErrorCodes.INVALID_REQUEST,
        "restricted room creation requires visibility, roomKind, and members",
      ),
    };
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
    return {
      restricted,
      error: errorShape(
        ErrorCodes.INVALID_REQUEST,
        "restricted rooms use the server-owned private execution policy",
      ),
    };
  }
  return { restricted, error: null };
}
