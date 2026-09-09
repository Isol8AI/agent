import * as sessionStore from "openclaw/plugin-sdk/session-store-runtime";
import { afterEach, describe, expect, it, vi } from "vitest";
import { isRestrictedMemorySession } from "./private-room.js";
import { filterMemorySearchHitsBySessionVisibility } from "./session-search-visibility.js";
import { searchHit } from "./session-search-visibility.test-support.js";
import { asOpenClawConfig } from "./tools.test-helpers.js";

vi.mock("openclaw/plugin-sdk/session-store-runtime", async (importOriginal) => {
  const actual = await importOriginal<typeof import("openclaw/plugin-sdk/session-store-runtime")>();
  return { ...actual, getSessionEntry: vi.fn() };
});

afterEach(() => vi.mocked(sessionStore.getSessionEntry).mockReset());

describe("private room memory boundary", () => {
  const cfg = asOpenClawConfig({});
  const sessionKey = "agent:main:room";

  it("reads the exact current session instead of caching an earlier public decision", () => {
    vi.mocked(sessionStore.getSessionEntry)
      .mockReturnValueOnce({ sessionId: "room", updatedAt: 1 })
      .mockReturnValueOnce({ sessionId: "room", updatedAt: 1, visibility: "restricted" });
    expect(isRestrictedMemorySession({ cfg, sessionKey })).toBe(false);
    expect(isRestrictedMemorySession({ cfg, sessionKey })).toBe(true);
    expect(sessionStore.getSessionEntry).toHaveBeenLastCalledWith(
      expect.objectContaining({ sessionKey, agentId: "main", readConsistency: "latest" }),
    );
  });

  it("denies global memory-only hits and cross-session hits before corpus visibility shortcuts", async () => {
    vi.mocked(sessionStore.getSessionEntry).mockReturnValue({
      sessionId: "room",
      updatedAt: 1,
      visibility: "restricted",
    });
    const hits = [
      searchHit("memory/global.md", "memory", "global secret"),
      searchHit("sessions/other.jsonl", "sessions", "other room secret"),
    ];
    await expect(
      filterMemorySearchHitsBySessionVisibility({
        cfg,
        requesterSessionKey: sessionKey,
        sandboxed: true,
        hits,
      }),
    ).resolves.toEqual([]);
  });
});
