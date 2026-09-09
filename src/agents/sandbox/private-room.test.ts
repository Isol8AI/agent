import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  PRIVATE_ROOM_CAPABILITIES,
  privateRoomPolicyForEntry,
  resolvePrivateRoomSessionRoot,
} from "../../config/sessions/private-room-policy.js";
import type {
  InternalSessionEntry,
  PrivateRoomExecutionPolicy,
} from "../../config/sessions/types.js";
import type { OpenClawConfig } from "../../config/types.openclaw.js";
import { resolveAgentDir, resolveAgentWorkspaceDir } from "../agent-scope.js";
import { resolveSandboxConfigForAgent } from "./config.js";
import { createPrivateRoomSandbox } from "./private-room.js";

function policy(root: string, sessionId: string): PrivateRoomExecutionPolicy {
  return {
    isolationSubject: { type: "session", sessionId },
    sandbox: "required",
    workspaceAccess: "none",
    sessionRoot: path.join(root, sessionId),
    toolPolicyVersion: "private-room-v1",
    allowedCapabilities: PRIVATE_ROOM_CAPABILITIES,
  };
}
describe("private room file capability sandbox", () => {
  it("keeps the durable room root stable when agent and workspace directories change", () => {
    const stateDir = path.join(os.tmpdir(), "durable-openclaw-state");
    const sessionId = "room-session";
    const configs = [
      {
        agents: {
          entries: {
            main: { agentDir: "/ephemeral/agent-a", workspace: "/ephemeral/workspace-a" },
          },
        },
      },
      {
        agents: {
          entries: {
            main: { agentDir: "/ephemeral/agent-b", workspace: "/ephemeral/workspace-b" },
          },
        },
      },
    ] satisfies OpenClawConfig[];

    for (const cfg of configs) {
      const root = resolvePrivateRoomSessionRoot({ agentId: "main", sessionId, stateDir });
      expect(root).toBe(path.join(stateDir, "private-rooms", "main", sessionId));
      expect(path.relative(resolveAgentDir(cfg, "main"), root)).toMatch(/^\.\./u);
      expect(path.relative(resolveAgentWorkspaceDir(cfg, "main"), root)).toMatch(/^\.\./u);
    }
  });

  it("rejects path-shaped private room components", () => {
    expect(() =>
      resolvePrivateRoomSessionRoot({ agentId: "../main", sessionId: "room" }),
    ).toThrow();
    expect(() =>
      resolvePrivateRoomSessionRoot({ agentId: "main", sessionId: "../room" }),
    ).toThrow();
  });

  it("keeps two rooms from one creator in distinct roots and rejects host or sibling files and shell", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "private-rooms-"));
    try {
      const cfg = resolveSandboxConfigForAgent({});
      const a = await createPrivateRoomSandbox({
        cfg,
        sessionKey: "agent:main:a",
        policy: policy(root, "a"),
      });
      const b = await createPrivateRoomSandbox({
        cfg,
        sessionKey: "agent:main:b",
        policy: policy(root, "b"),
      });
      expect(a.runtimeId).not.toBe(b.runtimeId);
      expect(a.workspaceDir).not.toBe(b.workspaceDir);
      expect(a.agentWorkspaceDir).toBe(a.workspaceDir);
      expect(a.workspaceAccess).toBe("none");
      await fs.writeFile(path.join(b.workspaceDir, "secret"), "sibling secret");
      await fs.symlink(b.workspaceDir, path.join(a.workspaceDir, "escape"));
      for (const filePath of [
        path.join(b.workspaceDir, "secret"),
        "../b/secret",
        "escape/secret",
        "/etc/passwd",
      ]) {
        await expect(a.fsBridge!.readFile({ filePath })).rejects.toThrow();
      }
      await expect(
        a.backend!.buildExecSpec({ command: "pwd", env: {}, usePty: false }),
      ).rejects.toThrow("does not permit shell");
      expect(a.browser).toBeUndefined();
      for (const name of [
        "sessions_send",
        "message",
        "memory_search",
        "memory_get",
        "browser",
        "exec",
        "new_plugin_tool",
      ]) {
        expect(a.tools.allow).not.toContain(name);
      }
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });
  it("fails closed on missing, mutable, unsupported, or mismatched persisted policy", () => {
    const entry: InternalSessionEntry = {
      sessionId: "a",
      updatedAt: 1,
      visibility: "restricted",
      privateRoomExecutionPolicy: policy("/tmp/rooms", "a"),
    };
    const resolved = privateRoomPolicyForEntry(entry)!;
    expect(Object.isFrozen(resolved)).toBe(true);
    expect(Object.isFrozen(resolved.allowedCapabilities)).toBe(true);
    expect(() => privateRoomPolicyForEntry({ ...entry, sessionId: "b" })).toThrow("policy");
    expect(() =>
      privateRoomPolicyForEntry({ ...entry, privateRoomExecutionPolicy: undefined }),
    ).toThrow("policy");
    expect(() =>
      privateRoomPolicyForEntry({
        ...entry,
        privateRoomExecutionPolicy: { ...resolved, allowedCapabilities: ["tool:exec"] },
      }),
    ).toThrow("policy");
  });
});
