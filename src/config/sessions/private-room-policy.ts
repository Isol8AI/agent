import path from "node:path";
import { normalizeAgentIdStrict } from "@openclaw/normalization-core/agent-id";
import { resolveSessionAgentId } from "../../agents/agent-scope.js";
import { resolveStateDir } from "../paths.js";
import type { OpenClawConfig } from "../types.openclaw.js";
import { resolveSessionStorePathCore } from "./paths.js";
import { resolveSessionEntry } from "./session-accessor.sqlite-entry.js";
import type { InternalSessionEntry, PrivateRoomExecutionPolicy } from "./types.js";

export const PRIVATE_ROOM_CAPABILITIES = Object.freeze([
  "tool:read",
  "tool:write",
  "tool:edit",
  "tool:apply_patch",
  "gateway:sessions.files.list",
  "gateway:sessions.files.get",
  "gateway:sessions.files.set",
  "gateway:sessions.files.reveal",
  "gateway:sessions.execution.dispatch",
]);

const PRIVATE_ROOM_SESSION_ID_RE = /^[a-z0-9][a-z0-9_-]{0,127}$/iu;

export function resolvePrivateRoomSessionRoot(params: {
  agentId: string;
  sessionId: string;
  stateDir?: string;
}): string {
  const agentId = normalizeAgentIdStrict(params.agentId);
  if (
    !agentId.ok ||
    agentId.value !== params.agentId ||
    !PRIVATE_ROOM_SESSION_ID_RE.test(params.sessionId)
  ) {
    throw new Error("Private room storage identity is invalid");
  }
  return path.join(
    path.resolve(params.stateDir ?? resolveStateDir()),
    "private-rooms",
    agentId.value,
    params.sessionId,
  );
}

/** Missing or malformed private policy never falls back to an agent workspace. */
export function privateRoomPolicyForEntry(
  entry: InternalSessionEntry | undefined,
): PrivateRoomExecutionPolicy | undefined {
  if (entry?.visibility !== "restricted") {
    return undefined;
  }
  const policy = entry.privateRoomExecutionPolicy;
  if (
    !policy ||
    policy.isolationSubject?.type !== "session" ||
    policy.isolationSubject.sessionId !== entry.sessionId ||
    policy.sandbox !== "required" ||
    policy.workspaceAccess !== "none" ||
    policy.toolPolicyVersion !== "private-room-v1" ||
    typeof policy.sessionRoot !== "string" ||
    !path.isAbsolute(policy.sessionRoot) ||
    path.basename(policy.sessionRoot) !== entry.sessionId ||
    !Array.isArray(policy.allowedCapabilities) ||
    policy.allowedCapabilities.some((capability) => !PRIVATE_ROOM_CAPABILITIES.includes(capability))
  ) {
    throw new Error("Private room execution policy is unavailable or unsupported");
  }
  return Object.freeze({
    ...policy,
    isolationSubject: Object.freeze({ ...policy.isolationSubject }),
    allowedCapabilities: Object.freeze([...policy.allowedCapabilities]),
  });
}

export function resolvePrivateRoomPolicy(params: {
  cfg?: OpenClawConfig;
  sessionKey?: string;
  agentId?: string;
}) {
  if (!params.sessionKey) {
    return undefined;
  }
  const agentId = resolveSessionAgentId({
    config: params.cfg,
    sessionKey: params.sessionKey,
    agentId: params.agentId,
  });
  const session = resolveSessionEntry(
    {
      agentId,
      sessionKey: params.sessionKey,
      storePath: resolveSessionStorePathCore(params.cfg?.session?.store, { agentId }),
      clone: false,
    },
    { readOnly: true },
  );
  return privateRoomPolicyForEntry(session.existing);
}
