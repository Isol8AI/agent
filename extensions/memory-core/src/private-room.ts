import { resolveSessionAgentIdStrict } from "openclaw/plugin-sdk/agent-scope-runtime";
import type { OpenClawConfig } from "openclaw/plugin-sdk/memory-core-host-runtime-core";
import { getSessionEntry, resolveStorePath } from "openclaw/plugin-sdk/session-store-runtime";

/** The room transcript is model context; global memory has no room-scoped corpus in v1. */
export function isRestrictedMemorySession(params: {
  cfg: OpenClawConfig;
  agentId?: string;
  sessionKey?: string;
}): boolean {
  if (!params.sessionKey) {
    return false;
  }
  const agentId = resolveSessionAgentIdStrict({
    config: params.cfg,
    agentId: params.agentId,
    sessionKey: params.sessionKey,
  });
  return (
    getSessionEntry({
      agentId,
      sessionKey: params.sessionKey,
      storePath: resolveStorePath(params.cfg.session?.store, { agentId }),
      readConsistency: "latest",
    })?.visibility === "restricted"
  );
}
