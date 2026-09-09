import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { prepareShellSourceRedactor } from "./redact-shell-source.js";
import { redactToolPayloadTextWithConfig } from "./redact.js";
import { registerSecretValueForRedaction } from "./secret-redaction-registry.js";
import { resetSecretRedactionRegistryForTest } from "./secret-redaction-registry.test-support.js";
let redact: Awaited<ReturnType<typeof prepareShellSourceRedactor>>;
beforeAll(async () => {
  redact = await prepareShellSourceRedactor();
});
afterEach(resetSecretRedactionRegistryForTest);
describe("owned shell header source", () => {
  it.each(["$TOKEN", "\${TOKEN}", "\${TOKEN:-}", "$(printenv TOKEN)"])(
    "preserves whole expanding %s",
    (value) => {
      for (const header of [
        "Authorization: Bearer",
        "Proxy-Authorization: Basic",
        "X-Api-Key:",
        "X-OpenClaw-Token=",
        "x-pomerium-jwt-assertion:",
        "X-Auth-Token:",
        "x-goog-api-key:",
      ]) {
        const source = 'curl -H "' + header + " " + value + '" https://example.invalid';
        expect(redact(source)).toBe(source);
        expect(redact(redact(source))).toBe(source);
      }
    },
  );
  it.each([
    "\${TOKEN:-literal}",
    "$(cat /tmp/token)",
    "$(printenv TOKEN EXTRA)",
    "$TOKEN-suffix",
    "\\$TOKEN",
    "literal credential with spaces",
    "$(echo $(printenv TOKEN))",
  ])("fully masks unsafe %s", (value) => {
    const source = 'curl -H "Authorization: Bearer ' + value + '" https://example.invalid';
    expect(redact(source)).toBe('curl -H "Authorization: Bearer ***" https://example.invalid');
  });
  it.each(["curl -H 'X-Api-Key: $TOKEN'", 'echo "X-Api-Key: $TOKEN"', 'curl "X-Api-Key: $TOKEN"'])(
    "does not preserve literal or non-header ownership %s",
    (source) => {
      expect(redact(source)).not.toContain("$TOKEN");
    },
  );
  it("keeps custom and registered masking ahead of source exceptions", () => {
    const source = 'curl -H "X-Api-Key: $TOKEN"';
    expect(redact(source, { redactPatterns: ["\\$TOKEN"] })).not.toContain("$TOKEN");
    registerSecretValueForRedaction("$TOKEN");
    expect(redact(source)).not.toContain("$TOKEN");
  });
  it("leaves diagnostics and parse failures conservative", () => {
    expect(redactToolPayloadTextWithConfig("X-Api-Key: $TOKEN")).not.toContain("$TOKEN");
    expect(redact('curl -H "X-Api-Key: $TOKEN')).not.toContain("$TOKEN");
    expect(redact(" ".repeat(140_000) + 'curl -H "X-Api-Key: $TOKEN"')).not.toContain("$TOKEN");
  });
  it("reparses offsets after custom edits and Unicode", () => {
    const source = 'echo "😀 remove-me"; curl -H "X-Api-Key: $TOKEN"';
    expect(redact(source, { redactPatterns: ["remove-me"] })).toContain(
      'curl -H "X-Api-Key: $TOKEN"',
    );
  });
});
