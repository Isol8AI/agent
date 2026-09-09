import path from "node:path";
import { describe, expect, it } from "vitest";
import { resolveAgentDir } from "../../agents/agent-scope-config.js";
import { normalizeAgentId } from "../../routing/session-key.js";
import { resolveSkillCollectionBackupRoot } from "./collection-paths.js";
import { resolveWorkshopSkillsDir } from "./skills-root.js";

describe("durable Workshop owner roots", () => {
  it("isolates shared workspaces and ignores external agentDir overrides", () => {
    const env = { OPENCLAW_STATE_DIR: "/durable/owner" };
    const config = {
      agents: {
        entries: {
          alpha: { workspace: "/shared", agentDir: "/ephemeral/alpha" },
          beta: { workspace: "/shared", agentDir: "/ephemeral/beta" },
        },
      },
    };
    for (const agentId of ["alpha", "beta", "ALPHA", "../beta"]) {
      const root = path.join("/durable/owner/skill-workshop/agents", normalizeAgentId(agentId));
      expect(resolveWorkshopSkillsDir(config, agentId, env)).toBe(path.join(root, "skills"));
      expect(resolveSkillCollectionBackupRoot(config, agentId, env)).toBe(
        path.join(root, "collection-backups"),
      );
    }
    expect(resolveWorkshopSkillsDir(config, "alpha", env)).not.toBe(
      resolveWorkshopSkillsDir(config, "beta", env),
    );
    expect(resolveAgentDir(config, "alpha", env)).toBe("/ephemeral/alpha");
  });
});
