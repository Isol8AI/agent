import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { runInNewContext } from "node:vm";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { CdpSendFn } from "./cdp.helpers.js";
import * as navigationGuard from "./navigation-guard.js";
import { createPageViaPlaywright } from "./pw-session-actions.js";
import {
  resolveSessionStateConfig,
  resolveProfileSessionStatePath,
  restoreSessionState,
  snapshotSessionState,
} from "./session-state-store.js";

vi.mock("./pw-session-actions.js", () => ({ createPageViaPlaywright: vi.fn() }));
const nativeAssertNavigation = navigationGuard.assertBrowserNavigationAllowed;
const navigation = { cdpUrl: "http://127.0.0.1:18800" };

let tmpDir: string;

beforeEach(async () => {
  vi.spyOn(navigationGuard, "assertBrowserNavigationAllowed").mockResolvedValue(undefined);
  vi.mocked(createPageViaPlaywright)
    .mockReset()
    .mockResolvedValue({ targetId: "t1", title: "", url: "https://example.com", type: "page" });
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "session-state-"));
});

afterEach(async () => {
  vi.restoreAllMocks();
  await fs.rm(tmpDir, { recursive: true, force: true });
});

type Handler = (params?: Record<string, unknown>, sessionId?: string) => unknown;

/** Build a CdpSendFn from a per-method handler map; unknown methods throw. */
function mockSend(handlers: Record<string, Handler>): CdpSendFn {
  return async (method, params, sessionId) => {
    const handler = handlers[method];
    if (!handler) {
      throw new Error(`unexpected CDP method: ${method}`);
    }
    return handler(params, sessionId);
  };
}

describe("snapshotSessionState", () => {
  it("does not attribute a navigated document's storage to its former origin", async () => {
    const outPath = path.join(tmpDir, "state.json");
    const send = mockSend({
      "Storage.getCookies": () => ({ cookies: [] }),
      "Target.getTargets": () => ({
        targetInfos: [{ type: "page", targetId: "t1", url: "https://a.example" }],
      }),
      "Target.attachToTarget": () => ({ sessionId: "s1" }),
      "Runtime.evaluate": (params) => ({
        result: {
          value: runInNewContext(
            String(params?.expression),
            {
              location: { origin: "https://b.example" },
              localStorage: { token: "b-secret" },
            },
            { timeout: 100 },
          ),
        },
      }),
      "Target.detachFromTarget": () => ({}),
    });
    expect(await snapshotSessionState(send, outPath)).toEqual({ cookies: 0, origins: 0 });
    expect(await fs.readFile(outPath, "utf8")).not.toContain("b-secret");
  });

  it.each(["closed", "failed", "empty"])(
    "retains unobserved storage but honors authoritative empty reads (%s)",
    async (mode) => {
      const filePath = path.join(tmpDir, "state.json");
      await fs.writeFile(
        filePath,
        JSON.stringify({
          version: 1,
          cookies: [],
          origins: [{ origin: "https://example.com", localStorage: { token: "retained" } }],
        }),
      );
      const origin = "https://example.com";
      let writes = 0;
      const restoreSend = mockSend({
        "Target.attachToTarget": () => ({ sessionId: "s1" }),
        "Runtime.evaluate": (params) => ({
          result: {
            value: runInNewContext(
              String(params?.expression),
              {
                location: { origin },
                localStorage: {
                  setItem: () => {
                    writes++;
                  },
                },
              },
              { timeout: 100 },
            ),
          },
        }),
        "Target.closeTarget": () => ({}),
      });
      expect((await restoreSessionState(restoreSend, filePath, navigation))?.origins).toBe(1);
      const snapshotSend = mockSend({
        "Storage.getCookies": () => ({ cookies: [] }),
        "Target.getTargets": () => ({
          targetInfos: mode === "closed" ? [] : [{ type: "page", targetId: "t1", url: origin }],
        }),
        "Target.attachToTarget": () => ({ sessionId: "s1" }),
        "Runtime.evaluate": () => {
          if (mode === "failed") {
            throw new Error("document unavailable");
          }
          return { result: { value: { origin, localStorage: {} } } };
        },
        "Target.detachFromTarget": () => ({}),
      });
      await snapshotSessionState(snapshotSend, filePath);
      const next = await restoreSessionState(restoreSend, filePath, navigation);
      expect(next?.origins).toBe(mode === "empty" ? 0 : 1);
      expect(writes).toBe(mode === "empty" ? 1 : 2);
    },
  );

  it("isolates two canonical profiles across snapshot and restart", async () => {
    const basePath = path.join(tmpDir, "state.json");
    for (const profile of ["openclaw", "work"]) {
      await snapshotSessionState(
        mockSend({
          "Storage.getCookies": () => ({
            cookies: [{ name: "sid", value: profile, domain: "example.com", path: "/" }],
          }),
          "Target.getTargets": () => ({ targetInfos: [] }),
        }),
        resolveProfileSessionStatePath(basePath, profile),
      );
    }
    expect(resolveProfileSessionStatePath(basePath, "openclaw")).toBe(basePath);
    for (const profile of ["openclaw", "work"]) {
      const restored: unknown[] = [];
      await restoreSessionState(
        mockSend({
          "Storage.setCookies": (params) => {
            restored.push(params?.cookies);
            return {};
          },
        }),
        resolveProfileSessionStatePath(basePath, profile),
      );
      expect(restored).toEqual([
        [{ name: "sid", value: profile, domain: "example.com", path: "/" }],
      ]);
    }
  });
  it("writes a v1 JSON with cookies and per-origin localStorage", async () => {
    const cookies = [{ name: "sid", value: "abc", domain: "example.com", path: "/" }];
    const send = mockSend({
      "Storage.getCookies": () => ({ cookies }),
      "Target.getTargets": () => ({
        targetInfos: [
          { type: "page", targetId: "t1", url: "https://example.com/dash" },
          // same origin as t1 → must be deduped (one origin entry, one attach)
          { type: "page", targetId: "t2", url: "https://example.com/other" },
          // non-http origins are skipped
          { type: "page", targetId: "t3", url: "chrome://newtab/" },
          { type: "page", targetId: "t4", url: "about:blank" },
          // non-page targets are skipped
          { type: "service_worker", targetId: "t5", url: "https://sw.example.com/" },
        ],
      }),
      "Target.attachToTarget": (params) => ({ sessionId: `sess-${String(params?.targetId)}` }),
      "Runtime.evaluate": () => ({
        result: { value: { origin: "https://example.com", localStorage: { token: "xyz", n: 5 } } },
      }),
      "Target.detachFromTarget": () => ({}),
    });

    const outPath = path.join(tmpDir, "browser-state", "state.json");
    const result = await snapshotSessionState(send, outPath);

    expect(result).toEqual({ cookies: 1, origins: 1 });

    const written = JSON.parse(await fs.readFile(outPath, "utf8"));
    expect(written.version).toBe(1);
    expect(typeof written.savedAt).toBe("string");
    expect(written.cookies).toEqual(cookies);
    // non-string localStorage values (n:5) are dropped
    expect(written.origins).toEqual([
      { origin: "https://example.com", localStorage: { token: "xyz" } },
    ]);
  });

  it("leaves no partial file or temp file when the write cannot be committed", async () => {
    const send = mockSend({
      "Storage.getCookies": () => ({ cookies: [] }),
      "Target.getTargets": () => ({ targetInfos: [] }),
    });

    // A directory at the target path makes the tmp→target rename fail.
    const dirTarget = path.join(tmpDir, "state-as-dir");
    await fs.mkdir(dirTarget);

    await expect(snapshotSessionState(send, dirTarget)).rejects.toBeDefined();

    const leftovers = (await fs.readdir(tmpDir)).filter((f) => f.endsWith(".tmp"));
    expect(leftovers).toEqual([]);
    expect((await fs.stat(dirTarget)).isDirectory()).toBe(true);
  });
});

describe("restoreSessionState", () => {
  it("rejects a strict-policy persisted private origin before creating a page", async () => {
    vi.mocked(navigationGuard.assertBrowserNavigationAllowed).mockImplementation(
      nativeAssertNavigation,
    );
    const filePath = path.join(tmpDir, "state.json");
    await fs.writeFile(
      filePath,
      JSON.stringify({
        version: 1,
        cookies: [],
        origins: [{ origin: "http://169.254.169.254", localStorage: { token: "never-send" } }],
      }),
    );
    const send = vi.fn<CdpSendFn>();
    expect(
      await restoreSessionState(send, filePath, {
        ...navigation,
        ssrfPolicy: { dangerouslyAllowPrivateNetwork: false },
      }),
    ).toEqual({ cookies: 0, origins: 0 });
    expect(createPageViaPlaywright).not.toHaveBeenCalled();
    expect(send).not.toHaveBeenCalled();
  });

  it("checks the expected origin in the same evaluation that writes secrets", async () => {
    const filePath = path.join(tmpDir, "state.json");
    await fs.writeFile(
      filePath,
      JSON.stringify({
        version: 1,
        cookies: [],
        origins: [{ origin: "https://example.com", localStorage: { token: "a-secret" } }],
      }),
    );
    const setItem = vi.fn();
    let reads = 0;
    const close = vi.fn(() => ({}));
    const send = mockSend({
      "Target.attachToTarget": () => ({ sessionId: "s1" }),
      "Runtime.evaluate": (params) => ({
        result: {
          value: runInNewContext(
            String(params?.expression),
            {
              location: { origin: reads++ === 0 ? "https://example.com" : "https://evil.example" },
              localStorage: { setItem },
            },
            { timeout: 100 },
          ),
        },
      }),
      "Target.closeTarget": close,
    });
    expect((await restoreSessionState(send, filePath, navigation))?.origins).toBe(0);
    expect(setItem).not.toHaveBeenCalled();
    expect(close).toHaveBeenCalled();
  });
  it("returns null when the snapshot file is absent (no CDP calls)", async () => {
    const calls: string[] = [];
    const send: CdpSendFn = async (method) => {
      calls.push(method);
      return {};
    };
    const result = await restoreSessionState(send, path.join(tmpDir, "missing.json"));
    expect(result).toBeNull();
    expect(calls).toEqual([]);
  });

  it("returns null on a version mismatch (no CDP calls)", async () => {
    const calls: string[] = [];
    const send: CdpSendFn = async (method) => {
      calls.push(method);
      return {};
    };
    const filePath = path.join(tmpDir, "state.json");
    await fs.writeFile(filePath, JSON.stringify({ version: 999, cookies: [], origins: [] }));
    const result = await restoreSessionState(send, filePath);
    expect(result).toBeNull();
    expect(calls).toEqual([]);
  });

  it("restores cookies and per-origin localStorage from a v1 file", async () => {
    const cookies = [{ name: "sid", value: "abc", domain: "example.com", path: "/" }];
    const filePath = path.join(tmpDir, "state.json");
    await fs.writeFile(
      filePath,
      JSON.stringify({
        version: 1,
        savedAt: new Date().toISOString(),
        cookies,
        origins: [{ origin: "https://example.com", localStorage: { token: "xyz" } }],
      }),
    );

    let setCookiesArg: unknown;
    let setItemsExpr: string | undefined;
    const send = mockSend({
      "Storage.setCookies": (params) => {
        setCookiesArg = params?.cookies;
        return {};
      },
      "Target.createTarget": () => ({ targetId: "t1" }),
      "Target.attachToTarget": () => ({ sessionId: "s1" }),
      "Runtime.evaluate": (params) => {
        const expr = typeof params?.expression === "string" ? params.expression : "";
        if (expr === "location.origin") {
          return { result: { value: "https://example.com" } };
        }
        setItemsExpr = expr;
        return { result: { value: 1 } };
      },
      "Target.closeTarget": () => ({}),
    });

    const result = await restoreSessionState(send, filePath, navigation);
    expect(result).toEqual({ cookies: 1, origins: 1 });
    expect(setCookiesArg).toEqual(cookies);
    expect(setItemsExpr).toContain("token");
    expect(setItemsExpr).toContain("localStorage.setItem");
    expect(createPageViaPlaywright).toHaveBeenCalledWith({
      ...navigation,
      url: "https://example.com",
    });
  });

  it("normalizes a session cookie's expires:-1 so Chrome stores it (not drops it)", async () => {
    // Storage.getCookies returns expires:-1 for session cookies; Storage.setCookies
    // reads -1 as a 1969 expiry and silently drops the cookie. The param must OMIT
    // `expires` (CDP's session-cookie form) instead, or every login cookie is lost.
    const cookies = [
      {
        name: "session_sid",
        value: "s",
        domain: "e.com",
        path: "/",
        expires: -1,
        size: 20,
        session: true,
        partitionKeyOpaque: false,
      },
      { name: "persist", value: "p", domain: "e.com", path: "/", expires: 9999999999 },
    ];
    const filePath = path.join(tmpDir, "state.json");
    await fs.writeFile(
      filePath,
      JSON.stringify({
        version: 1,
        savedAt: new Date().toISOString(),
        cookies,
        origins: [],
      }),
    );

    let setCookiesArg: Array<Record<string, unknown>> | undefined;
    const send = mockSend({
      "Storage.setCookies": (params) => {
        setCookiesArg = params?.cookies as Array<Record<string, unknown>>;
        return {};
      },
    });

    const result = await restoreSessionState(send, filePath);
    expect(result).toEqual({ cookies: 2, origins: 0 });
    const session = setCookiesArg?.find((c) => c.name === "session_sid") as Record<string, unknown>;
    const persist = setCookiesArg?.find((c) => c.name === "persist") as Record<string, unknown>;
    // session cookie: expires removed (Chrome would drop it if left at -1)
    expect(session).toBeDefined();
    expect("expires" in session).toBe(false);
    // read-only Cookie fields stripped so the bulk setCookies is not rejected
    expect("size" in session).toBe(false);
    expect("session" in session).toBe(false);
    expect("partitionKeyOpaque" in session).toBe(false);
    // persistent cookie: real future expiry preserved
    expect(persist?.expires).toBe(9999999999);
  });

  it("continues past a rejected cookie (per-item tolerance)", async () => {
    const cookies = [
      { name: "good", value: "1", domain: "e.com", path: "/" },
      { name: "bad", value: "2", domain: "e.com", path: "/" },
    ];
    const filePath = path.join(tmpDir, "state.json");
    await fs.writeFile(
      filePath,
      JSON.stringify({ version: 1, savedAt: "x", cookies, origins: [] }),
    );

    let bulkTried = false;
    const send: CdpSendFn = async (method, params) => {
      if (method !== "Storage.setCookies") {
        throw new Error(`unexpected CDP method: ${method}`);
      }
      const arr = (params as { cookies?: Array<{ name?: string }> })?.cookies ?? [];
      if (arr.length > 1) {
        bulkTried = true;
        throw new Error("bulk rejected");
      }
      if (arr[0]?.name === "bad") {
        throw new Error("bad cookie");
      }
      return {};
    };

    const result = await restoreSessionState(send, filePath);
    expect(bulkTried).toBe(true);
    expect(result).toEqual({ cookies: 1, origins: 0 });
  });
});

describe("resolveSessionStateConfig", () => {
  it("defaults to disabled with a home-expanded path when no config is set", () => {
    const resolved = resolveSessionStateConfig(undefined);
    expect(resolved.enabled).toBe(false);
    expect(resolved.intervalMs).toBe(60_000);
    expect(resolved.path.endsWith("/.openclaw/browser-state/state.json")).toBe(true);
    expect(resolved.path.startsWith("~")).toBe(false);
  });

  it("enables on presence and honors interval + custom path", () => {
    const resolved = resolveSessionStateConfig({ intervalSeconds: 30, path: "/data/state.json" });
    expect(resolved.enabled).toBe(true);
    expect(resolved.intervalMs).toBe(30_000);
    expect(resolved.path).toBe("/data/state.json");
  });

  it("floors the interval to avoid wake-spam and honors enabled:false", () => {
    expect(resolveSessionStateConfig({ intervalSeconds: 1 }).intervalMs).toBe(15_000);
    expect(resolveSessionStateConfig({ enabled: false }).enabled).toBe(false);
  });
});
