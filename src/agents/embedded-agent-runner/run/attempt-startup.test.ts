import { beforeEach, describe, expect, it, vi } from "vitest";
import { withPrivateRoomExecution } from "../../private-room-execution.js";
import { prepareEmbeddedSkills } from "../skill-runtime.js";
import type { EmbeddedRunAttemptParams } from "./types.js";

const mocks = vi.hoisted(() => ({
  applySkillEnvOverrides: vi.fn(),
  mapSandboxSkillEntriesForPrompt: vi.fn(),
}));

vi.mock("../../../skills/runtime/env-overrides.js", () => ({
  applySkillEnvOverrides: mocks.applySkillEnvOverrides,
  applySkillEnvOverridesFromSnapshot: vi.fn(),
}));

vi.mock("../../../skills/runtime/embedded-run-entries.js", () => ({
  resolveEmbeddedRunSkillEntries: vi.fn(() => ({
    shouldLoadSkillEntries: true,
    skillEntries: [],
    loadSkillEntries: vi.fn(() => []),
  })),
}));

vi.mock("../../../skills/loading/workspace-skill-prompt.js", () => ({
  resolveSkillsPrompt: vi.fn(() => "skills prompt"),
}));

vi.mock("../sandbox-skills.js", () => ({
  createSandboxPromptEntryLoader: vi.fn(
    ({ loadEntries }: { loadEntries: () => unknown[] }) => loadEntries,
  ),
  resolveSandboxSkillRuntimeInputs: vi.fn(() => ({
    skillsEligibility: undefined,
    skillsPromptWorkspaceDir: "/tmp/workspace",
    skillsSnapshot: undefined,
    skillsWorkspaceDir: "/tmp/workspace",
    workspaceOnly: false,
  })),
  mapSandboxSkillEntriesForPrompt: mocks.mapSandboxSkillEntriesForPrompt,
  mapSandboxSkillUsagePaths: vi.fn(() => []),
}));

describe("prepareEmbeddedSkills", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not load global skill context or apply skill environment in a private room", () => {
    const prepared = withPrivateRoomExecution(
      {
        agentId: "main",
        rootExecutionId: "root",
        runId: "run",
        hopCount: 0,
        sessionKey: "agent:main:room",
        sessionId: "room",
        inputMessageId: "input",
        assertCurrent: () => {},
        close: () => {},
      },
      () =>
        prepareEmbeddedSkills({
          includeCodeModeSkills: true,
          attempt: { config: {} } as EmbeddedRunAttemptParams,
          effectiveWorkspace: "/tmp/room",
          sandbox: null,
          sessionAgentId: "main",
        }),
    );
    expect(prepared.skillsPrompt).toBe("");
    expect(prepared.codeModeSkills).toEqual([]);
    expect(mocks.applySkillEnvOverrides).not.toHaveBeenCalled();
    expect(mocks.mapSandboxSkillEntriesForPrompt).not.toHaveBeenCalled();
  });

  it("restores environment overrides when later preparation fails", () => {
    const restore = vi.fn();
    mocks.applySkillEnvOverrides.mockReturnValue(restore);
    mocks.mapSandboxSkillEntriesForPrompt.mockImplementation(() => {
      throw new Error("skill prompt mapping failed");
    });

    expect(() =>
      prepareEmbeddedSkills({
        includeCodeModeSkills: true,
        attempt: { config: {} } as EmbeddedRunAttemptParams,
        effectiveWorkspace: "/tmp/workspace",
        sandbox: null,
        sessionAgentId: "main",
      }),
    ).toThrow("skill prompt mapping failed");
    expect(restore).toHaveBeenCalledOnce();
  });

  it("does not load skills or apply their environment during settled finalization", () => {
    const prepared = prepareEmbeddedSkills({
      includeCodeModeSkills: true,
      attempt: { operation: "settled-tool-finalization" } as EmbeddedRunAttemptParams,
      effectiveWorkspace: "/tmp/workspace",
      sandbox: null,
      sessionAgentId: "main",
    });

    expect(prepared.skillsPrompt).toBe("");
    expect(prepared.skillsSnapshotForRun).toBeUndefined();
    expect(mocks.applySkillEnvOverrides).not.toHaveBeenCalled();
    expect(mocks.mapSandboxSkillEntriesForPrompt).not.toHaveBeenCalled();
  });
});
