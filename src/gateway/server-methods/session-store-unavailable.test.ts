import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  loadGatewaySessionStoreReads,
  readGatewaySessionStore,
  SessionLookupUnavailableError,
} from "../session-utils-store-read.js";
const reads = vi.hoisted(() => ({ exact: vi.fn(), batch: vi.fn(), list: vi.fn() }));
vi.mock("../../config/sessions/session-accessor.js", () => ({
  loadExactSessionEntryCandidates: reads.exact,
  loadExactSessionEntryCandidatesReadOnlyBatch: reads.batch,
  listSessionEntriesCore: reads.list,
  listSessionEntriesReadOnly: reads.list,
}));
beforeEach(() => vi.resetAllMocks());
describe("session store failures are unavailable, not empty", () => {
  it.each(["exact", "list"] as const)(
    "rejects failed %s reads without caching emptiness",
    (mode) => {
      reads[mode].mockImplementation(() => {
        throw new Error("synthetic locked database");
      });
      const read = {
        storePath: "/synthetic/sessions.json",
        options: { ...(mode === "exact" ? { exactKeys: ["a"] } : {}), cache: new Map() },
      };
      expect(() => readGatewaySessionStore(read)).toThrow(SessionLookupUnavailableError);
      expect(read.options.cache.size).toBe(0);
    },
  );
  it("rejects a failed batched store", () => {
    reads.batch.mockReturnValue([{ ok: false, error: new Error("synthetic corrupt database") }]);
    expect(() =>
      loadGatewaySessionStoreReads([
        { storePath: "/synthetic/sessions.json", options: { exactKeys: ["a"] } },
      ]),
    ).toThrow(SessionLookupUnavailableError);
  });
  it("preserves successful empty reads", () => {
    reads.exact.mockReturnValue([]);
    expect(
      readGatewaySessionStore({
        storePath: "/synthetic/sessions.json",
        options: { exactKeys: ["missing"] },
      }),
    ).toEqual({});
  });
});
