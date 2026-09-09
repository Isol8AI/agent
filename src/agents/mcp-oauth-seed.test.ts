import fs from "node:fs/promises";
import path from "node:path";
import { withTempHome as withBaseTempHome } from "openclaw/plugin-sdk/test-env";
import { expect, it } from "vitest";
import { resolveStateDir } from "../config/paths.js";
import { closeOpenClawStateDatabaseForTest } from "../state/openclaw-state-db.js";
import {
  operatorMcpOAuthIdentity,
  requesterMcpOAuthIdentity,
  type McpOAuthIdentity,
} from "./mcp-oauth-identity.js";
import { createMcpOAuthClientProvider } from "./mcp-oauth-provider.js";
import { readMcpOAuthStore } from "./mcp-oauth-store.js";
import { resolveMcpOAuthAccessToken } from "./mcp-oauth.js";
const REMOTE_IDENTITY = operatorMcpOAuthIdentity("Remote Docs", "https://mcp.example.com/mcp");
function requesterIdentity(serverName: string, serverUrl: string, requesterSenderId: string) {
  return requesterMcpOAuthIdentity(serverName, serverUrl, {
    messageChannel: "telegram",
    agentAccountId: "bot",
    requesterSenderId,
  });
}
async function saveAccessToken(identity: McpOAuthIdentity, accessToken: string): Promise<void> {
  await createMcpOAuthClientProvider({ identity }).saveTokens({
    access_token: accessToken,
    token_type: "Bearer",
    expires_in: 3600,
  });
}
async function withTempHome<T>(
  run: (home: string) => T | Promise<T>,
  options: Parameters<typeof withBaseTempHome>[1],
): Promise<T> {
  return withBaseTempHome(async (home) => {
    const previousStateDir = process.env.OPENCLAW_STATE_DIR;
    process.env.OPENCLAW_STATE_DIR = path.join(home, ".openclaw");
    closeOpenClawStateDatabaseForTest();
    try {
      return await run(home);
    } finally {
      closeOpenClawStateDatabaseForTest();
      if (previousStateDir === undefined) {
        delete process.env.OPENCLAW_STATE_DIR;
      } else {
        process.env.OPENCLAW_STATE_DIR = previousStateDir;
      }
    }
  }, options);
}

it("imports external operator seeds at resolution without crossing server or requester identities", async () => {
  await withTempHome(
    async () => {
      const seedDir = path.join(resolveStateDir(), "mcp-oauth");
      await fs.mkdir(seedDir, { recursive: true });
      const other = operatorMcpOAuthIdentity("Distinct Operator Server", REMOTE_IDENTITY.serverUrl);
      await saveAccessToken(REMOTE_IDENTITY, "fixture-stale-token");
      await saveAccessToken(other, "fixture-distinct-operator-token");
      const seed = {
        ...readMcpOAuthStore(REMOTE_IDENTITY.storeKey),
        tokens: { access_token: "fixture-new-seed-token", token_type: "Bearer", expires_in: 3600 },
      };
      const seedPath = path.join(seedDir, REMOTE_IDENTITY.storeKey + ".json");
      await fs.writeFile(seedPath, JSON.stringify(seed));
      expect(await resolveMcpOAuthAccessToken({ identity: other })).toBe(
        "fixture-distinct-operator-token",
      );
      expect(await fs.readFile(seedPath, "utf8")).toContain("fixture-new-seed-token");
      expect(await resolveMcpOAuthAccessToken({ identity: REMOTE_IDENTITY })).toBe(
        "fixture-new-seed-token",
      );
      await expect(fs.access(seedPath)).rejects.toHaveProperty("code", "ENOENT");
      const requester = requesterIdentity(
        REMOTE_IDENTITY.serverName,
        REMOTE_IDENTITY.serverUrl,
        "alice",
      );
      expect(
        await resolveMcpOAuthAccessToken({ identity: requester, allowMissingToken: true }),
      ).not.toBe("fixture-new-seed-token");
      await fs.writeFile(seedPath, "{malformed");
      expect(await resolveMcpOAuthAccessToken({ identity: REMOTE_IDENTITY })).toBe(
        "fixture-new-seed-token",
      );
      expect(await fs.readFile(seedPath, "utf8")).toBe("{malformed");
    },
    {
      prefix: "openclaw-oauth-seed-",
      skipSessionCleanup: true,
      env: { OPENCLAW_CONFIG_PATH: undefined, OPENCLAW_STATE_DIR: undefined },
    },
  );
});

it("keeps the same operator identity isolated across two owner state roots", async () => {
  const options = {
    prefix: "owner-oauth-seed-",
    skipSessionCleanup: true,
    env: { OPENCLAW_CONFIG_PATH: undefined, OPENCLAW_STATE_DIR: undefined },
  };
  await withTempHome(async () => {
    await saveAccessToken(REMOTE_IDENTITY, "owner-a-old");
    const seed = {
      ...readMcpOAuthStore(REMOTE_IDENTITY.storeKey),
      tokens: {
        access_token: "owner-a-seed",
        token_type: "Bearer",
        expires_in: 3600,
      },
    };
    const aPath = path.join(resolveStateDir(), "mcp-oauth", REMOTE_IDENTITY.storeKey + ".json");
    await fs.mkdir(path.dirname(aPath), { recursive: true });
    await fs.writeFile(aPath, JSON.stringify(seed));
    await withTempHome(async () => {
      expect(
        await resolveMcpOAuthAccessToken({ identity: REMOTE_IDENTITY, allowMissingToken: true }),
      ).toBeUndefined();
      const bPath = path.join(resolveStateDir(), "mcp-oauth", REMOTE_IDENTITY.storeKey + ".json");
      expect(bPath).not.toBe(aPath);
      await fs.mkdir(path.dirname(bPath), { recursive: true });
      await fs.writeFile(
        bPath,
        JSON.stringify({ ...seed, tokens: { ...seed.tokens, access_token: "owner-b-seed" } }),
      );
      const requester = requesterIdentity(
        REMOTE_IDENTITY.serverName,
        REMOTE_IDENTITY.serverUrl,
        "alice",
      );
      expect(
        await resolveMcpOAuthAccessToken({ identity: requester, allowMissingToken: true }),
      ).toBeUndefined();
      expect(await fs.readFile(bPath, "utf8")).toContain("owner-b-seed");
      expect(await resolveMcpOAuthAccessToken({ identity: REMOTE_IDENTITY })).toBe("owner-b-seed");
      expect(await fs.readFile(aPath, "utf8")).toContain("owner-a-seed");
    }, options);
    expect(await resolveMcpOAuthAccessToken({ identity: REMOTE_IDENTITY })).toBe("owner-a-seed");
    await expect(fs.access(aPath)).rejects.toHaveProperty("code", "ENOENT");
  }, options);
});
