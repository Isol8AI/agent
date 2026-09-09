/** Active execution owners, not a transcript or a durable queue. */
const executions = new Set<{ assertCurrent: () => void; abort: () => void }>();

export function registerPrivateRoomExecution(owner: {
  assertCurrent: () => void;
  abort: () => void;
}): () => void {
  executions.add(owner);
  return () => {
    executions.delete(owner);
  };
}

export function revokePrivateRoomExecutions(): void {
  for (const owner of executions) {
    try {
      owner.assertCurrent();
    } catch {
      owner.abort();
      executions.delete(owner);
    }
  }
}
