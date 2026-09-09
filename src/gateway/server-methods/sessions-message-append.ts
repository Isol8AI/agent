import { createHash } from "node:crypto";
import {
  ErrorCodes,
  errorShape,
  validateSessionMessageAppendParams,
  type SessionMessageAppendResult,
  type SessionMemberIdentity,
} from "../../../packages/gateway-protocol/src/index.js";
import { privateRoomExecutionForRun } from "../../agents/private-room-execution.js";
import { resolveSessionWorkStartError } from "../../config/sessions/lifecycle.js";
import {
  loadSessionEntryReadOnly,
  persistSessionTranscriptTurn,
  readActiveTranscriptEntryAnchor,
  recordSessionParticipant,
} from "../../config/sessions/session-accessor.js";
import { readCommittedTranscriptMessageSequence } from "../../config/sessions/session-accessor.sqlite-transcript-sequences.js";
import { isSessionMember } from "../../config/sessions/session-sharing-store.js";
import { getAgentRunContext } from "../../infra/agent-run-registry.js";
import {
  OPENCLAW_TRANSCRIPT_ARTIFACT_API,
  OPENCLAW_TRANSCRIPT_ARTIFACT_PROVIDER,
} from "../../shared/transcript-only-openclaw-assistant.js";
import { SessionMutationAuthorizationChangedError } from "../session-mutation-authorization-error.js";
import {
  authorizeIncognitoSessionTarget,
  authorizeSessionMessageAppendTarget,
  hiddenSessionNotFound,
  isGatewayAdmin,
  resolveSessionSharingTarget,
} from "../session-sharing-policy.js";
import type { GatewayRequestHandler } from "./types.js";
import { assertValidParams } from "./validation.js";

/** Append one authenticated contribution without admitting execution or delivery. */
export const appendSessionMessage: GatewayRequestHandler = async ({
  params,
  client,
  context,
  respond,
  sessionMutationCommitGuard,
  sessionMutationAuthorization,
  signal,
}) => {
  if (
    !assertValidParams(
      params,
      validateSessionMessageAppendParams,
      "sessions.message.append",
      respond,
    )
  ) {
    return;
  }
  const request = structuredClone(params);
  const cfg = context.getRuntimeConfig();
  const runtime = client?.internal?.agentRuntimeIdentity;
  const profileId = client?.authenticatedUserProfile?.profileId;
  const sender: SessionMemberIdentity | undefined = runtime
    ? { type: "agent", id: runtime.agentId }
    : profileId && !client?.internal?.syntheticClient
      ? { type: "profile", id: profileId }
      : undefined;
  function reject(message: string): never {
    throw new SessionMutationAuthorizationChangedError(
      errorShape(ErrorCodes.INVALID_REQUEST, message),
    );
  }
  try {
    if (!sender?.id.trim()) {
      reject("message append requires an authenticated profile or admitted agent run");
    }
    // Capture producer identity once; every use rechecks its live owner.
    const identity = sender;
    const assertAuthor = () => {
      signal?.throwIfAborted();
      if (client?.invalidated || client?.connectionSignal?.aborted) {
        reject("message append connection is no longer active");
      }
      if (runtime) {
        const run = getAgentRunContext(runtime.operationalRunInstance.runId);
        if (
          context.validateAgentRuntimeApprovalAuthority?.(runtime) !== true ||
          run?.agentId !== identity.id ||
          run.sessionKey !== runtime.sessionKey
        ) {
          reject("message append agent run is no longer active");
        }
      } else if (client?.authenticatedUserProfile?.profileId !== identity.id) {
        reject("message append profile changed");
      }
    };
    assertAuthor();
    const target = resolveSessionSharingTarget({ cfg, sessionKey: request.sessionKey });
    if (!target) {
      throw new SessionMutationAuthorizationChangedError(hiddenSessionNotFound(request.sessionKey));
    }
    const scope = {
      agentId: target.agentId,
      storePath: target.storePath,
      sessionKey: target.storeKey,
      sessionId: request.expectedSessionId,
    };
    const assertTarget = () => {
      sessionMutationCommitGuard?.();
      sessionMutationAuthorization?.assertCurrent();
      assertAuthor();
      const currentCfg = context.getRuntimeConfig();
      const current = resolveSessionSharingTarget({
        cfg: currentCfg,
        sessionKey: request.sessionKey,
      });
      if (
        !current ||
        current.storeKey !== scope.sessionKey ||
        current.storePath !== scope.storePath
      ) {
        reject("message append session changed");
      }
      const unavailable = resolveSessionWorkStartError(scope.sessionKey, current.entry, {
        expectedSessionId: request.expectedSessionId,
        allowPendingWorkspace: true,
      });
      if (unavailable) {
        reject(unavailable);
      }
      const accessError =
        authorizeIncognitoSessionTarget({
          client,
          sessionKey: scope.sessionKey,
          target: current,
        }) ?? authorizeSessionMessageAppendTarget({ cfg: currentCfg, client, target: current });
      if (accessError) {
        throw new SessionMutationAuthorizationChangedError(accessError);
      }
      // An autonomous run's operator grants and presentation profile never confer room membership.
      if (
        current.entry.visibility === "restricted" &&
        !isGatewayAdmin(client) &&
        !isSessionMember(scope, identity)
      ) {
        reject("message append requires room membership");
      }
      if (runtime && current.entry.visibility === "restricted") {
        const execution = privateRoomExecutionForRun(runtime.operationalRunInstance);
        if (
          !execution ||
          execution.sessionId !== scope.sessionId ||
          execution.sessionKey !== current.canonicalKey ||
          runtime.sessionKey !== current.canonicalKey
        ) {
          reject("private room result requires the exact authenticated execution");
        }
      }
      if (
        request.replyToId &&
        !readActiveTranscriptEntryAnchor({ ...scope, entryId: request.replyToId })
      ) {
        reject("replyToId must identify a message in this session");
      }
      return true;
    };
    assertTarget();
    // ponytail: one result per run/target; add owner-issued result IDs when a run needs multiple results.
    // Model/client input cannot mint another result identity.
    const producerKey = runtime
      ? [
          "agent-result",
          runtime.operationalRunInstance.instanceId,
          runtime.operationalRunInstance.runId,
        ]
      : ["profile", identity.id, request.idempotencyKey];
    const idempotencyKey = `session-message:${createHash("sha256").update(JSON.stringify(producerKey)).digest("hex")}`;
    const timestamp = Date.now();
    const message = {
      role: runtime ? "assistant" : "user",
      content: [{ type: "text", text: request.text }],
      timestamp,
      idempotencyKey,
      ...(runtime
        ? {
            api: OPENCLAW_TRANSCRIPT_ARTIFACT_API,
            provider: OPENCLAW_TRANSCRIPT_ARTIFACT_PROVIDER,
            model: "room-result",
            stopReason: "stop",
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
          }
        : {}),
      __openclaw: {
        senderId: identity.id,
        senderIdentity: identity,
        ...(request.replyToId ? { replyToId: request.replyToId } : {}),
        ...(request.mentions ? { mentions: request.mentions } : {}),
        // Compare producer-owned bytes even when storage redaction maps different text to the same value.
        messageAppendDigest: createHash("sha256")
          .update(
            JSON.stringify([
              request.text,
              request.replyToId,
              request.mentions?.map(({ type, id }) => [type, id]),
            ]),
          )
          .digest("hex"),
      },
    };
    const committed = await persistSessionTranscriptTurn(scope, {
      expectedSessionId: request.expectedSessionId,
      config: cfg,
      messages: [{ message, now: timestamp, shouldAppendInTransaction: assertTarget }],
      touchSessionEntry: true,
      updateMode: "inline",
      // The canonical owner invokes this only after commit, before publishing session.message.
      onMessageCommitted: (receipt) => {
        try {
          if (
            receipt.appended &&
            loadSessionEntryReadOnly({ ...scope, readConsistency: "latest" })?.sessionId ===
              scope.sessionId
          ) {
            recordSessionParticipant(scope, { identity, promptedAt: timestamp });
          }
        } catch {
          // Best-effort activity must not suppress a committed message's receipt or publication.
        }
      },
    });
    const receipt = committed.messages[0];
    if (!receipt) {
      reject("message append session changed before commit");
    }
    const messageSeq =
      readCommittedTranscriptMessageSequence(receipt) ??
      (receipt.anchor ? receipt.anchor.activeMessagePosition + 1 : undefined);
    if (messageSeq === undefined) {
      throw new Error("committed message sequence is unavailable");
    }
    respond(true, {
      sessionKey: scope.sessionKey,
      sessionId: scope.sessionId,
      messageId: receipt.messageId,
      messageSeq,
      ...(receipt.effectiveParentId ? { effectiveParentId: receipt.effectiveParentId } : {}),
      appended: receipt.appended,
    } satisfies SessionMessageAppendResult);
  } catch (error) {
    if (error instanceof SessionMutationAuthorizationChangedError) {
      respond(false, undefined, error.error);
    } else if (error instanceof Error && error.name === "TranscriptTurnAdmissionConflictError") {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.INVALID_REQUEST, "message append idempotency conflict"),
      );
    } else {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.UNAVAILABLE, "message append could not be committed"),
      );
    }
  }
};
