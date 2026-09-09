/**
 * Gateway device-auth regression tests.
 */
import { describe, expect, it } from "vitest";
import {
  buildDeviceAuthPayload,
  buildDeviceAuthPayloadV3,
  buildDeviceAuthPayloadV4,
  normalizeDeviceMetadataForAuth,
} from "./device-auth.js";

describe("device-auth payload vectors", () => {
  it.each([
    {
      name: "builds canonical v2 payloads",
      build: () =>
        buildDeviceAuthPayload({
          deviceId: "dev-1",
          clientId: "openclaw-macos",
          clientMode: "ui",
          role: "operator",
          scopes: ["operator.admin", "operator.read"],
          signedAtMs: 1_700_000_000_000,
          token: null,
          nonce: "nonce-abc",
        }),
      expected:
        "v2|dev-1|openclaw-macos|ui|operator|operator.admin,operator.read|1700000000000||nonce-abc",
    },
    {
      name: "builds canonical v3 payloads",
      build: () =>
        buildDeviceAuthPayloadV3({
          deviceId: "dev-1",
          clientId: "openclaw-macos",
          clientMode: "ui",
          role: "operator",
          scopes: ["operator.admin", "operator.read"],
          signedAtMs: 1_700_000_000_000,
          token: "tok-123",
          nonce: "nonce-abc",
          platform: "  IOS  ",
          deviceFamily: "  iPhone  ",
        }),
      expected:
        "v3|dev-1|openclaw-macos|ui|operator|operator.admin,operator.read|1700000000000|tok-123|nonce-abc|ios|iphone",
    },
    {
      name: "binds a trusted broker profile in canonical v4 payloads",
      build: () =>
        buildDeviceAuthPayloadV4({
          deviceId: "dev-3",
          clientId: "gateway-client",
          clientMode: "backend",
          role: "operator",
          scopes: ["operator.read", "operator.write"],
          signedAtMs: 1_700_000_000_002,
          token: "tok-456",
          nonce: "nonce-ghi",
          platform: "Linux",
          trustedBrokerProfileId: "profile-human-1",
        }),
      expected:
        "v4|5:dev-3|14:gateway-client|7:backend|8:operator|1:2|13:operator.read|14:operator.write|13:1700000000002|7:tok-456|9:nonce-ghi|5:linux|0:|15:profile-human-1",
    },
    {
      name: "keeps empty metadata slots in v3 payloads",
      build: () =>
        buildDeviceAuthPayloadV3({
          deviceId: "dev-2",
          clientId: "openclaw-ios",
          clientMode: "ui",
          role: "operator",
          scopes: ["operator.read"],
          signedAtMs: 1_700_000_000_001,
          nonce: "nonce-def",
        }),
      expected: "v3|dev-2|openclaw-ios|ui|operator|operator.read|1700000000001||nonce-def||",
    },
  ])("$name", ({ build, expected }) => {
    expect(build()).toBe(expected);
  });

  it("keeps delimiter placement and scope boundaries distinct in v4 payloads", () => {
    const base = {
      deviceId: "device",
      clientId: "gateway-client",
      clientMode: "backend",
      role: "operator",
      signedAtMs: 1_700_000_000_002,
      platform: "linux",
      trustedBrokerProfileId: "profile-human-1",
    };

    expect(
      buildDeviceAuthPayloadV4({ ...base, scopes: ["scope"], token: "tok|nonce", nonce: "x" }),
    ).not.toBe(
      buildDeviceAuthPayloadV4({ ...base, scopes: ["scope"], token: "tok", nonce: "nonce|x" }),
    );
    expect(
      buildDeviceAuthPayloadV4({
        ...base,
        scopes: ["scope,a", "scope-b"],
        token: "token",
        nonce: "nonce",
      }),
    ).not.toBe(
      buildDeviceAuthPayloadV4({
        ...base,
        scopes: ["scope", "a,scope-b"],
        token: "token",
        nonce: "nonce",
      }),
    );
  });

  it.each([
    { input: "  İOS  ", expected: "İos" },
    { input: "  MAC  ", expected: "mac" },
    { input: undefined, expected: "" },
  ])("normalizes metadata %j", ({ input, expected }) => {
    expect(normalizeDeviceMetadataForAuth(input)).toBe(expected);
  });
});
