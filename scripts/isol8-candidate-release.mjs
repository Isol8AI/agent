import { execFileSync } from "node:child_process";
import { appendFileSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
export const BASE = "1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7";
export const IMAGE = "ghcr.io/thelightbulbcompany/agent";
export function validateInputs(tag, sha) {
  if (
    typeof tag !== "string" ||
    tag !== tag.trim() ||
    typeof sha !== "string" ||
    sha !== sha.trim()
  ) {
    throw new Error("Invalid candidate tag or release SHA");
  }
  if (!/^v2026\.9\.3-isol8\.[1-9][0-9]*$/.test(tag ?? "") || !/^[a-f0-9]{40}$/.test(sha ?? "")) {
    throw new Error("Invalid candidate tag or release SHA");
  }
}
export function identity(
  tag,
  sha,
  git = (...args) => execFileSync("git", args, { encoding: "utf8" }).trim(),
  version = JSON.parse(readFileSync("package.json", "utf8")).version,
) {
  validateInputs(tag, sha);
  const ref = "refs/tags/" + tag;
  if (git("cat-file", "-t", ref) !== "tag") {
    throw new Error("Candidate must be annotated");
  }
  const object = git("rev-parse", ref);
  if (git("rev-parse", ref + "^{commit}") !== sha || git("rev-parse", "HEAD") !== sha) {
    throw new Error("Candidate source identity mismatch");
  }
  git("merge-base", "--is-ancestor", BASE, sha);
  if (version !== "2026.9.3") {
    throw new Error("Wrong package version");
  }
  return { tag, sha, object, version: tag.slice(1) };
}
export function classifyManifest(status, body, digest, expected) {
  if (expected !== undefined) {
    if (status !== 200 || !/^sha256:[a-f0-9]{64}$/.test(expected) || digest !== expected) {
      throw new Error("Published digest mismatch");
    }
    return digest;
  }
  if (
    status === 404 &&
    Array.isArray(body?.errors) &&
    body.errors.length > 0 &&
    body.errors.every((error) => error?.code === "MANIFEST_UNKNOWN")
  ) {
    return undefined;
  }
  throw new Error(status === 200 ? "Candidate tag already exists" : "Registry absence is unproven");
}
async function github(route) {
  const response = await fetch("https://api.github.com/" + route, {
    headers: {
      Authorization: "Bearer " + process.env.GH_TOKEN,
      Accept: "application/vnd.github+json",
    },
    signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) {
    throw new Error("GitHub identity lookup failed: " + response.status);
  }
  return response.json();
}
async function verifyRemote(candidate) {
  const repository = process.env.GITHUB_REPOSITORY;
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository ?? "")) {
    throw new Error("Invalid repository");
  }
  const ref = await github("repos/" + repository + "/git/ref/tags/" + candidate.tag);
  if (ref.object?.type !== "tag" || ref.object.sha !== candidate.object) {
    throw new Error("Remote tag changed");
  }
  const tag = await github("repos/" + repository + "/git/tags/" + candidate.object);
  if (tag.object?.type !== "commit" || tag.object.sha !== candidate.sha) {
    throw new Error("Remote peeled source changed");
  }
}
async function registry(version, expected) {
  const pkg = await github("orgs/TheLightbulbCompany/packages/container/agent");
  if (pkg.visibility !== "private") {
    throw new Error("GHCR package is not private");
  }
  const auth = await fetch(
    "https://ghcr.io/token?service=ghcr.io&scope=repository:thelightbulbcompany/agent:pull",
    {
      headers: {
        Authorization:
          "Basic " +
          Buffer.from(process.env.GITHUB_ACTOR + ":" + process.env.GH_TOKEN).toString("base64"),
      },
      signal: AbortSignal.timeout(30000),
    },
  );
  if (!auth.ok) {
    throw new Error("Registry authentication failed: " + auth.status);
  }
  const { token } = await auth.json();
  if (typeof token !== "string" || !token) {
    throw new Error("Missing registry token");
  }
  const response = await fetch(
    "https://ghcr.io/v2/thelightbulbcompany/agent/manifests/" + version,
    {
      headers: {
        Authorization: "Bearer " + token,
        Accept:
          "application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json",
      },
      signal: AbortSignal.timeout(30000),
    },
  );
  const body = response.status === 404 ? await response.json() : undefined;
  return classifyManifest(
    response.status,
    body,
    response.headers.get("docker-content-digest"),
    expected,
  );
}
async function main() {
  const candidate = identity(process.env.CANDIDATE_TAG, process.env.RELEASE_SHA);
  await verifyRemote(candidate);
  if (process.argv[2] === "identity") {
    appendFileSync(
      process.env.GITHUB_OUTPUT,
      "version=" +
        candidate.version +
        "\nobject=" +
        candidate.object +
        "\ncreated=" +
        new Date().toISOString() +
        "\n",
    );
  } else if (process.argv[2] === "absent") {
    if (candidate.object !== process.env.TAG_OBJECT) {
      throw new Error("Local tag changed");
    }
    await registry(candidate.version);
  } else if (process.argv[2] === "proof") {
    if (
      candidate.object !== process.env.TAG_OBJECT ||
      !/^sha256:[a-f0-9]{64}$/.test(process.env.BUILD_DIGEST ?? "")
    ) {
      throw new Error("Missing or changed build identity");
    }
    const digest = await registry(candidate.version, process.env.BUILD_DIGEST);
    appendFileSync(
      process.env.GITHUB_STEP_SUMMARY,
      "Candidate: " +
        candidate.tag +
        "\n\nTag object: " +
        candidate.object +
        "\n\nPeeled commit: " +
        candidate.sha +
        "\n\nUpstream: v2026.9.3 / " +
        BASE +
        "\n\nUTC build: " +
        process.env.BUILD_CREATED +
        "\n\nRun: " +
        process.env.GITHUB_SERVER_URL +
        "/" +
        process.env.GITHUB_REPOSITORY +
        "/actions/runs/" +
        process.env.GITHUB_RUN_ID +
        "\n\nVerified image: " +
        IMAGE +
        ":" +
        candidate.version +
        "@" +
        digest +
        "\n",
    );
  } else {
    throw new Error("Unknown verification phase");
  }
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(
    /** @param {unknown} error */ (error) => {
      console.error(error instanceof Error ? error.message : String(error));
      process.exitCode = 1;
    },
  );
}
