import path from "node:path";
import type { OpenClawConfig } from "../../config/types.openclaw.js";
import { resolveWorkshopAgentRoot } from "./agent-root.js";

export function resolveWorkshopSkillsDir(
  _config: OpenClawConfig,
  agentId: string,
  env: NodeJS.ProcessEnv = process.env,
): string {
  return path.join(resolveWorkshopAgentRoot(agentId, env), "skills");
}

export function resolveWorkshopWatchRoots(config?: OpenClawConfig, agentId?: string) {
  return config && agentId
    ? [{ path: resolveWorkshopSkillsDir(config, agentId), source: "openclaw-workshop" }]
    : [];
}

export function createWorkshopWatcherKey(
  workspaceDir: string,
  params: { executionSkillsDir?: string; agentId?: string },
): string {
  return JSON.stringify([workspaceDir, params.executionSkillsDir, params.agentId]);
}
