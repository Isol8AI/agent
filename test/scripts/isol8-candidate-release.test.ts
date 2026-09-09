import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { parse } from "yaml";
import {
  BASE,
  classifyManifest,
  identity,
  validateInputs,
} from "../../scripts/isol8-candidate-release.mjs";
const tag = "v2026.9.3-isol8.1";
const sha = "a".repeat(40);
const object = "b".repeat(40);
function gitFixture(overrides: Record<string, string> = {}) {
  return vi.fn((...args: string[]) => {
    const key = args.join(" ");
    if (key === "merge-base --is-ancestor " + BASE + " " + sha) return "";
    return (
      overrides[key] ??
      (
        {
          ["cat-file -t refs/tags/" + tag]: "tag",
          ["rev-parse refs/tags/" + tag]: object,
          ["rev-parse refs/tags/" + tag + "^{commit}"]: sha,
          "rev-parse HEAD": sha,
        } as Record<string, string>
      )[key] ??
      ""
    );
  });
}
describe("private candidate identity gate", () => {
  it.each([
    "v2026.9.2-isol8.1",
    "v2026.9.3",
    "2026.9.3-isol8.1",
    "v2026.9.3-isol8.0",
    "latest",
    "v2026.9.3-isol8.1\n",
  ])("rejects invalid tag %s before reading Git", (candidate) => {
    const git = gitFixture();
    expect(() => identity(candidate, sha, git, "2026.9.3")).toThrow();
    expect(git).not.toHaveBeenCalled();
  });
  it.each(["abc123", "A".repeat(40), sha + "x", ""])("rejects noncanonical SHA %s", (candidate) => {
    expect(() => validateInputs(tag, candidate)).toThrow();
  });
  it("rejects lightweight tags", () =>
    expect(() =>
      identity(tag, sha, gitFixture({ ["cat-file -t refs/tags/" + tag]: "commit" }), "2026.9.3"),
    ).toThrow("annotated"));
  it.each(["rev-parse HEAD", "rev-parse refs/tags/" + tag + "^{commit}"])(
    "rejects mismatch at %s",
    (key) => {
      expect(() => identity(tag, sha, gitFixture({ [key]: "c".repeat(40) }), "2026.9.3")).toThrow(
        "mismatch",
      );
    },
  );
  it("rejects non-descendant source", () => {
    const git = gitFixture();
    const checked = (...args: string[]) => {
      if (args[0] === "merge-base") throw new Error("not descendant");
      return git(...args);
    };
    expect(() => identity(tag, sha, checked, "2026.9.3")).toThrow("not descendant");
  });
  it("rejects the wrong package version", () =>
    expect(() => identity(tag, sha, gitFixture(), "2026.9.2")).toThrow("package"));
  it("derives exactly one version from verified source", () =>
    expect(identity(tag, sha, gitFixture(), "2026.9.3")).toEqual({
      tag,
      sha,
      object,
      version: tag.slice(1),
    }));
});
describe("registry lookup fails closed", () => {
  it("accepts only positive manifest absence", () =>
    expect(
      classifyManifest(404, { errors: [{ code: "MANIFEST_UNKNOWN" }] }, null),
    ).toBeUndefined());
  it.each([200, 401, 403, 429, 500])("rejects HTTP %s", (status) =>
    expect(() => classifyManifest(status, {}, null)).toThrow(),
  );
  it.each([
    {},
    { errors: [] },
    { errors: [{ code: "UNAUTHORIZED" }] },
    { errors: [{ code: "NAME_UNKNOWN" }] },
  ])("rejects ambiguous 404 %j", (body) =>
    expect(() => classifyManifest(404, body, null)).toThrow(),
  );
  it("requires exact post-push digest equality", () => {
    const digest = "sha256:" + "d".repeat(64);
    expect(classifyManifest(200, undefined, digest, digest)).toBe(digest);
    expect(() => classifyManifest(200, undefined, "sha256:" + "e".repeat(64), digest)).toThrow();
    expect(() =>
      classifyManifest(404, { errors: [{ code: "MANIFEST_UNKNOWN" }] }, null, digest),
    ).toThrow();
    expect(() => classifyManifest(200, undefined, "", "")).toThrow();
  });
});
it("has one manual-only pinned private amd64 publisher and no upstream callers", () => {
  const workflow = parse(readFileSync(".github/workflows/docker-release.yml", "utf8"));
  expect(Object.keys(workflow.on)).toEqual(["workflow_dispatch"]);
  expect(Object.keys(workflow.on.workflow_dispatch.inputs)).toEqual(["tag", "release_sha"]);
  expect(workflow.concurrency).toEqual({
    group: "isol8-private-agent-publisher",
    "cancel-in-progress": false,
  });
  const steps = workflow.jobs.publish.steps;
  const uses = steps
    .filter((step: { uses?: string }) => step.uses)
    .map((step: { uses: string }) => step.uses);
  expect(uses).toEqual([
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
    "docker/setup-buildx-action@37fe631027851001ddb9b187196cc803df7f5f0e",
    "docker/login-action@dbcb813823bdd20940b903addbd779551569679f",
    "docker/build-push-action@53b7df96c91f9c12dcc8a07bcb9ccacbed38856a",
  ]);
  const build = steps.find((step: { id?: string }) => step.id === "build");
  expect(build.with).toMatchObject({
    platforms: "linux/amd64",
    push: true,
    sbom: true,
    provenance: "mode=max",
  });
  expect(build.with.tags).toBe(
    "ghcr.io/thelightbulbcompany/agent:${{ steps.candidate.outputs.version }}",
  );
  const identityIndex = steps.findIndex((step: { id?: string }) => step.id === "candidate");
  const loginIndex = steps.findIndex((step: { uses?: string }) =>
    step.uses?.startsWith("docker/login-action"),
  );
  expect(identityIndex).toBeLessThan(loginIndex);
  expect(existsSync(".github/workflows/docker-image-refresh.yml")).toBe(false);
  expect(existsSync(".github/workflows/openclaw-release-publish.yml")).toBe(false);
});
