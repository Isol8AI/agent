import path from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import type { HelloOk } from "../../packages/gateway-protocol/src/schema/frames.js";
import {
  GATEWAY_OWNER_PROFILE_ID,
  type UsersSelfResult,
} from "../../packages/gateway-protocol/src/schema/users.js";
import { useAutoCleanupTempDirTracker } from "../../test/helpers/temp-dir.js";
import { writeConfigFile } from "../config/config.js";
import { loadOrCreateDeviceIdentity } from "../infra/device-identity.js";
import { approveDevicePairing } from "../infra/device-pairing-approval.js";
import { revokeDeviceToken } from "../infra/device-pairing-tokens.js";
import { getPairedDevice, requestDevicePairing } from "../infra/device-pairing.js";
import { ensureProfileForEmail } from "../state/user-profiles.js";
import {
  BACKEND_GATEWAY_CLIENT,
  connectReq,
  createSignedDevice,
  installGatewayTestHooks,
  openWs,
  readConnectChallengeNonce,
  rpcReq,
  testState,
  withGatewayServer,
} from "./server.auth.test-helpers.js";

installGatewayTestHooks({ scope: "suite" });

const tempDirs = useAutoCleanupTempDirTracker(afterEach);
const TOKEN = "trusted-broker-test-token";
const SCOPES = ["operator.read", "operator.write"];

function identityPath(label: string): string {
  return path.join(tempDirs.make("openclaw-trusted-broker-"), `${label}.sqlite`);
}

async function configure(bindings?: Record<string, string>): Promise<void> {
  const auth = {
    mode: "token" as const,
    token: TOKEN,
    ...(bindings ? { trustedBrokerProfiles: bindings } : {}),
  };
  testState.gatewayAuth = auth;
  await writeConfigFile({ gateway: { auth } });
}

describe("token-authenticated trusted broker profiles", () => {
  test("binds one configured paired device to its signed profile without echoing auth input", async () => {
    const deviceIdentityPath = identityPath("valid");
    const device = loadOrCreateDeviceIdentity({ path: deviceIdentityPath });
    const profile = ensureProfileForEmail("human@example.com");
    await configure({ [device.deviceId]: profile.id });

    await withGatewayServer(async ({ port }) => {
      const ws = await openWs(port);
      try {
        const connected = await connectReq(ws, {
          token: TOKEN,
          trustedBrokerProfileId: profile.id,
          prePairDevice: true,
          scopes: SCOPES,
          client: BACKEND_GATEWAY_CLIENT,
          deviceIdentityPath,
        });
        expect(connected.ok, JSON.stringify(connected.error)).toBe(true);
        const hello = connected.payload as HelloOk;
        expect(hello.auth).not.toHaveProperty("token");
        expect(hello.auth).not.toHaveProperty("trustedBrokerProfileId");
        expect(hello.auth).not.toHaveProperty("deviceToken");
        expect(hello.auth).not.toHaveProperty("deviceTokens");
        expect(await rpcReq<UsersSelfResult>(ws, "users.self")).toMatchObject({
          ok: true,
          payload: { profile: { id: profile.id } },
        });
      } finally {
        ws.close();
      }
    });
  });

  test("rejects a mapped device without creating a missing pairing", async () => {
    const deviceIdentityPath = identityPath("missing-pairing");
    const device = loadOrCreateDeviceIdentity({ path: deviceIdentityPath });
    const profile = ensureProfileForEmail("missing-pairing@example.com");
    await configure({ [device.deviceId]: profile.id });

    await withGatewayServer(async ({ port }) => {
      const ws = await openWs(port);
      try {
        const connected = await connectReq(ws, {
          token: TOKEN,
          trustedBrokerProfileId: profile.id,
          prePairDevice: false,
          scopes: SCOPES,
          client: BACKEND_GATEWAY_CLIENT,
          deviceIdentityPath,
        });
        expect(connected.ok).toBe(false);
        expect(connected.error?.message).toContain("existing paired device");
        expect(await getPairedDevice(device.deviceId)).toBeNull();
      } finally {
        ws.close();
      }
    });
  });

  test("rejects a mapped device after its paired role token is revoked", async () => {
    const deviceIdentityPath = identityPath("revoked-pairing");
    const device = loadOrCreateDeviceIdentity({ path: deviceIdentityPath });
    const profile = ensureProfileForEmail("revoked-pairing@example.com");
    await configure({ [device.deviceId]: profile.id });

    await withGatewayServer(async ({ port }) => {
      const first = await openWs(port);
      try {
        const connected = await connectReq(first, {
          token: TOKEN,
          trustedBrokerProfileId: profile.id,
          prePairDevice: true,
          scopes: SCOPES,
          client: BACKEND_GATEWAY_CLIENT,
          deviceIdentityPath,
        });
        expect(connected.ok, JSON.stringify(connected.error)).toBe(true);
      } finally {
        first.close();
      }

      expect(
        await revokeDeviceToken({ deviceId: device.deviceId, role: "operator" }),
      ).toMatchObject({ ok: true });

      const second = await openWs(port);
      try {
        const connected = await connectReq(second, {
          token: TOKEN,
          trustedBrokerProfileId: profile.id,
          prePairDevice: false,
          scopes: SCOPES,
          client: BACKEND_GATEWAY_CLIENT,
          deviceIdentityPath,
        });
        expect(connected.ok).toBe(false);
        expect(connected.error?.message).toContain("existing paired device");
      } finally {
        second.close();
      }
    });
  });

  test.each([
    { name: "missing", asserted: undefined },
    { name: "mismatched", asserted: "profile-human-other" },
  ])("rejects a $name assertion for a configured device", async ({ asserted }) => {
    const deviceIdentityPath = identityPath(`invalid-${asserted ?? "missing"}`);
    const device = loadOrCreateDeviceIdentity({ path: deviceIdentityPath });
    const profile = ensureProfileForEmail(`${asserted ?? "missing"}@example.com`);
    await configure({ [device.deviceId]: profile.id });

    await withGatewayServer(async ({ port }) => {
      const ws = await openWs(port);
      try {
        const connected = await connectReq(ws, {
          token: TOKEN,
          ...(asserted ? { trustedBrokerProfileId: asserted } : {}),
          prePairDevice: true,
          scopes: SCOPES,
          client: BACKEND_GATEWAY_CLIENT,
          deviceIdentityPath,
        });
        expect(connected.ok).toBe(false);
        expect(connected.error?.message).toContain("trusted broker profile assertion");
      } finally {
        ws.close();
      }
    });
  });

  test("rejects an assertion added after an older device payload was signed", async () => {
    const deviceIdentityPath = identityPath("forged");
    const identity = loadOrCreateDeviceIdentity({ path: deviceIdentityPath });
    const profile = ensureProfileForEmail("forged@example.com");
    await configure({ [identity.deviceId]: profile.id });

    await withGatewayServer(async ({ port }) => {
      const ws = await openWs(port);
      try {
        const nonce = await readConnectChallengeNonce(ws);
        const { device } = await createSignedDevice({
          token: TOKEN,
          scopes: SCOPES,
          clientId: BACKEND_GATEWAY_CLIENT.id,
          clientMode: BACKEND_GATEWAY_CLIENT.mode,
          identityPath: deviceIdentityPath,
          nonce,
        });
        const pairing = await requestDevicePairing({
          deviceId: device.id,
          publicKey: device.publicKey,
          role: "operator",
          scopes: SCOPES,
          clientId: BACKEND_GATEWAY_CLIENT.id,
          clientMode: BACKEND_GATEWAY_CLIENT.mode,
          platform: BACKEND_GATEWAY_CLIENT.platform,
          silent: false,
        });
        await approveDevicePairing(pairing.request.requestId, { callerScopes: SCOPES });

        const connected = await connectReq(ws, {
          token: TOKEN,
          trustedBrokerProfileId: profile.id,
          scopes: SCOPES,
          client: BACKEND_GATEWAY_CLIENT,
          device,
          skipConnectChallengeNonce: true,
        });
        expect(connected.ok).toBe(false);
        expect(connected.error?.message).toContain("device signature");
      } finally {
        ws.close();
      }
    });
  });

  test("keeps ordinary token-authenticated devices on the gateway owner profile", async () => {
    const deviceIdentityPath = identityPath("ordinary");
    await configure();

    await withGatewayServer(async ({ port }) => {
      const ws = await openWs(port);
      try {
        const connected = await connectReq(ws, {
          token: TOKEN,
          prePairDevice: true,
          scopes: SCOPES,
          client: BACKEND_GATEWAY_CLIENT,
          deviceIdentityPath,
        });
        expect(connected.ok, JSON.stringify(connected.error)).toBe(true);
        expect(await rpcReq<UsersSelfResult>(ws, "users.self")).toMatchObject({
          ok: true,
          payload: { profile: { id: GATEWAY_OWNER_PROFILE_ID } },
        });
      } finally {
        ws.close();
      }
    });
  });
});
