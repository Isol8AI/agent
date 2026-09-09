import { lookup } from "node:dns";
import { afterEach, describe, expect, it, vi } from "vitest";
vi.mock("../config/config.js", () => ({
  getRuntimeConfig: () => ({ browser: { sessionState: { enabled: true } } }),
}));
vi.mock("./chrome.js", () => ({ getChromeWebSocketEndpoint: vi.fn() }));
vi.mock("./cdp.helpers.js", () => ({ withCdpSocket: vi.fn(async () => undefined) }));
import { withCdpSocket } from "./cdp.helpers.js";
import { getChromeWebSocketEndpoint } from "./chrome.js";
import type { RunningChrome } from "./chrome.js";
import { makeBrowserProfile, makeBrowserServerState } from "./server-context.test-harness.js";
import {
  restoreManagedBrowserSessionState,
  startBrowserSessionStateSnapshotTimer,
} from "./session-state-launch.js";
afterEach(() => {
  vi.clearAllMocks();
  vi.useRealTimers();
});
describe("session-state pinned CDP transport", () => {
  it.each(["restore", "snapshot"])("forwards the resolved URL and DNS lookup", async (run) => {
    vi.mocked(getChromeWebSocketEndpoint).mockResolvedValue({
      url: "ws://127.0.0.1:18800/devtools/browser/synthetic",
      lookup,
    });
    const state = makeBrowserServerState();
    if (run === "restore") {
      await restoreManagedBrowserSessionState({
        profile: makeBrowserProfile(),
        resolved: state.resolved,
      });
    } else {
      vi.useFakeTimers();
      state.profiles.set("openclaw", {
        profile: makeBrowserProfile(),
        running: { pid: 1 } as RunningChrome,
      });
      const stop = startBrowserSessionStateSnapshotTimer({ state });
      await vi.advanceTimersByTimeAsync(60_000);
      await stop();
    }
    expect(withCdpSocket).toHaveBeenCalledWith(
      "ws://127.0.0.1:18800/devtools/browser/synthetic",
      expect.any(Function),
      { commandTimeoutMs: 15_000, lookup },
    );
    // Native local-managed CDP intentionally scopes out page-navigation policy.
    expect(getChromeWebSocketEndpoint).toHaveBeenCalledWith(expect.any(String), 15_000, undefined);
  });
});
