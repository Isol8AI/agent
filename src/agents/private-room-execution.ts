import { AsyncLocalStorage } from "node:async_hooks";
import { isRecord } from "@openclaw/normalization-core/record-coerce";
import type { OperationalRunInstanceRef } from "./admitted-run-context.js";

export type PrivateRoomExecution = {
  readonly agentId: string;
  readonly rootExecutionId: string;
  readonly runId: string;
  readonly hopCount: number;
  readonly sessionKey: string;
  readonly sessionId: string;
  readonly inputMessageId: string;
  readonly assertCurrent: () => void;
  readonly close: () => void;
};
const current = new AsyncLocalStorage<PrivateRoomExecution>();
const runs = new Map<
  string,
  { instance: OperationalRunInstanceRef; execution: PrivateRoomExecution }
>();

export function withPrivateRoomExecution<T>(
  execution: PrivateRoomExecution,
  operation: () => T,
): T {
  execution.assertCurrent();
  if (!Number.isInteger(execution.hopCount) || execution.hopCount < 0 || execution.hopCount > 3) {
    throw new Error("Private room delegation exceeds the three-hop ceiling");
  }
  return current.run(execution, operation);
}
export function getPrivateRoomExecution(): PrivateRoomExecution | undefined {
  return current.getStore();
}
export function bindPrivateRoomRun(instance: OperationalRunInstanceRef): void {
  const execution = current.getStore();
  if (execution) {
    execution.assertCurrent();
    if (execution.runId !== instance.runId) {
      throw new Error("Private room run identity changed");
    }
    runs.set(instance.runId, { instance, execution });
  }
}
export function privateRoomExecutionForRun(
  instance: OperationalRunInstanceRef,
): PrivateRoomExecution | undefined {
  const bound = runs.get(instance.runId);
  const execution =
    bound?.instance.instanceId === instance.instanceId ? bound.execution : undefined;
  execution?.assertCurrent();
  return execution;
}
export function unbindPrivateRoomRun(instance: OperationalRunInstanceRef): void {
  if (runs.get(instance.runId)?.instance === instance) {
    runs.delete(instance.runId);
  }
}
export function assertPrivateRoomExecutionTarget(scope: {
  sessionKey?: string;
  sessionId?: string;
}): void {
  const execution = current.getStore();
  if (!execution) {
    return;
  }
  execution.assertCurrent();
  if (
    scope.sessionKey !== execution.sessionKey ||
    (scope.sessionId && scope.sessionId !== execution.sessionId)
  ) {
    throw new Error("Private room execution cannot access another session");
  }
}

/** Result identity is stamped after model/hook output, from the authenticated execution. */
export function stampPrivateRoomAssistant<T>(message: T): T {
  const execution = current.getStore();
  if (!execution || !isRecord(message) || message.role !== "assistant") {
    return message;
  }
  execution.assertCurrent();
  return {
    ...message,
    __openclaw: {
      ...(isRecord(message.__openclaw) ? message.__openclaw : {}),
      senderId: execution.agentId,
      senderIdentity: { type: "agent", id: execution.agentId },
    },
  };
}
