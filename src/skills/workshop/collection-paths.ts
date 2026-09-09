import path from "node:path";
import type { OpenClawConfig } from "../../config/types.openclaw.js";
import { resolveWorkshopAgentRoot } from "./agent-root.js";

export function resolveSkillCollectionBackupRoot(
  _config: OpenClawConfig,
  agentId: string,
  env?: NodeJS.ProcessEnv,
): string {
  return path.join(resolveWorkshopAgentRoot(agentId, env), "collection-backups");
}
