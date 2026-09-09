import { describe, expect, it } from "vitest";
import { assertPrivateRoomExecutionTarget, bindPrivateRoomRun, privateRoomExecutionForRun,
  unbindPrivateRoomRun, withPrivateRoomExecution, type PrivateRoomExecution } from "./private-room-execution.js";

describe("private room execution authority", () => {
  it("binds one authenticated root to an exact instance, session, and live owner", async () => {
    let active = true;
    const execution: PrivateRoomExecution = {
      agentId: "main", rootExecutionId: "root", runId: "run", hopCount: 3, sessionKey: "agent:main:room",
      sessionId: "room-instance", inputMessageId: "committed-human-input",
      assertCurrent: () => { if (!active) { throw new Error("revoked"); } },
      close: () => { active = false; },
    };
    const instance = { runId: "run", instanceId: "instance" };
    await withPrivateRoomExecution(execution, async () => {
      bindPrivateRoomRun(instance);
      expect(privateRoomExecutionForRun({ ...instance })).toBe(execution);
      expect(privateRoomExecutionForRun({ ...instance, instanceId: "replacement" })).toBeUndefined();
      expect(() => assertPrivateRoomExecutionTarget({ sessionKey: "agent:main:another" })).toThrow("another session");
      expect(() => assertPrivateRoomExecutionTarget({ sessionKey: execution.sessionKey, sessionId: "old" })).toThrow("another session");
      await Promise.resolve();
      active = false;
      expect(() => assertPrivateRoomExecutionTarget(execution)).toThrow("revoked");
      expect(() => privateRoomExecutionForRun(instance)).toThrow("revoked");
    });
    unbindPrivateRoomRun(instance);
    expect(privateRoomExecutionForRun(instance)).toBeUndefined();
  });
  it("rejects excessive and malformed hops before starting work", () => {
    for (const hopCount of [4, -1, 1.5, NaN]) {
      let called = false;
      expect(() => withPrivateRoomExecution({ agentId: "main", rootExecutionId: "root", runId: "run", hopCount,
        sessionKey: "room", sessionId: "instance", inputMessageId: "input", assertCurrent: () => {}, close: () => {} },
        () => { called = true; })).toThrow("three-hop");
      expect(called).toBe(false);
    }
  });
});
