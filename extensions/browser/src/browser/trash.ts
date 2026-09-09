import fs from "node:fs/promises";
/**
 * Trash helpers for data under the Browser-owned config subtree.
 */
import path from "node:path";
import { CONFIG_DIR } from "openclaw/plugin-sdk/text-utility-runtime";
import {
  resolveProfileSessionStatePath,
  resolveSessionStateConfig,
  type SessionStateConfig,
} from "./session-state-store.js";

/** Called only after the profile lifecycle has drained snapshot writers. */
export async function retireProfileSessionState(
  profileName: string,
  config?: SessionStateConfig,
): Promise<void> {
  const snapshotPath = path.resolve(
    resolveProfileSessionStatePath(resolveSessionStateConfig(config).path, profileName),
  );
  const stat = await fs.lstat(snapshotPath).catch((error: unknown) => {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      return undefined;
    }
    throw error;
  });
  if (!stat) {
    return;
  }
  if (!stat.isFile()) {
    throw new Error("Browser session snapshot is not a regular file");
  }
  const { movePathToTrash: moveExactPathToTrash } =
    await import("openclaw/plugin-sdk/browser-config");
  await moveExactPathToTrash(snapshotPath, { allowedRoots: [path.dirname(snapshotPath)] });
}

/** Moves a path to trash only when it lives under allowed Browser roots. */
export async function movePathToTrash(targetPath: string): Promise<string> {
  const { movePathToTrash: movePathToTrashWithAllowedRoots } =
    await import("openclaw/plugin-sdk/browser-config");
  return await movePathToTrashWithAllowedRoots(targetPath, {
    // Managed browser data follows OPENCLAW_STATE_DIR/OPENCLAW_CONFIG_PATH, which
    // may intentionally live outside the OS home. Limit authority to Browser's
    // owned subtree; fs-safe also checks target identity, realpaths, and symlinks.
    allowedRoots: [path.join(CONFIG_DIR, "browser")],
  });
}
