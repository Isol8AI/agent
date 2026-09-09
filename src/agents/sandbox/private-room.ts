import fs from "node:fs/promises";
import type { PrivateRoomExecutionPolicy } from "../../config/sessions/types.js";
import { runCommandWithTimeout } from "../../process/exec.js";
import { getPrivateRoomExecution } from "../private-room-execution.js";
import { createSandboxFsBridge } from "./fs-bridge.js";
import type { SandboxConfig, SandboxContext } from "./types.js";

/** File capabilities execute inside the founder container, with no room container or shell tool. */
export async function createPrivateRoomSandbox(params: {
  policy: PrivateRoomExecutionPolicy;
  sessionKey: string;
  cfg: SandboxConfig;
}): Promise<SandboxContext> {
  const { policy } = params;
  const execution = getPrivateRoomExecution();
  execution?.assertCurrent();
  if (execution && (
    execution.sessionId !== policy.isolationSubject.sessionId ||
    execution.sessionKey !== params.sessionKey
  )) {
    throw new Error("Private room sandbox requires the exact execution identity");
  }
  await fs.mkdir(policy.sessionRoot, { recursive: true, mode: 0o700 });
  execution?.assertCurrent();
  const backendId = "private-room-files-v1";
  const runtimeId = `private-room:${policy.isolationSubject.sessionId}`;
  const sandbox: SandboxContext = {
    enabled: true,
    required: true,
    backendId,
    sessionKey: params.sessionKey,
    workspaceDir: policy.sessionRoot,
    agentWorkspaceDir: policy.sessionRoot,
    workspaceAccess: "none",
    runtimeId,
    runtimeLabel: runtimeId,
    containerName: runtimeId,
    containerWorkdir: policy.sessionRoot,
    docker: { ...params.cfg.docker, binds: [], env: {}, network: "none" },
    tools: {
      allow: policy.allowedCapabilities
        .filter((cap) => cap.startsWith("tool:"))
        .map((cap) => cap.slice(5)),
    },
    browserAllowHostControl: false,
  };
  // Only the existing pinned filesystem plans can reach this transport. Do not expose
  // it as a backend exec capability: a future shell/browser policy needs a real sandbox.
  sandbox.fsBridge = createSandboxFsBridge({
    sandbox: {
      ...sandbox,
      backend: {
        async runShellCommand(command) {
          execution?.assertCurrent();
          command.signal?.throwIfAborted();
          const result = await runCommandWithTimeout(
            ["/bin/sh", "-c", command.script, "private-room-fs", ...(command.args ?? [])],
            {
              cwd: policy.sessionRoot,
              baseEnv: { PATH: "/usr/local/bin:/usr/bin:/bin", LANG: "C.UTF-8" },
              input: command.stdin,
              signal: command.signal,
              timeoutMs: 30_000,
              maxCombinedOutputBytes: 16 * 1024 * 1024,
            },
          );
          command.signal?.throwIfAborted();
          execution?.assertCurrent();
          if (result.code !== 0 && !command.allowFailure) {
            throw new Error("Private room filesystem operation failed");
          }
          return {
            stdout: Buffer.from(result.stdout),
            stderr: Buffer.from(result.stderr),
            code: result.code ?? 1,
          };
        },
      },
    },
  });
  const denyExec = async (): Promise<never> => {
    throw new Error("Private room policy does not permit shell execution");
  };
  sandbox.backend = {
    id: backendId,
    runtimeId,
    runtimeLabel: runtimeId,
    workdir: policy.sessionRoot,
    capabilities: { browser: false },
    buildExecSpec: denyExec,
    runShellCommand: denyExec,
  };
  return sandbox;
}
