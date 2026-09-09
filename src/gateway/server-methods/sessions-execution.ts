import { createHash } from "node:crypto";
import { asOptionalRecord } from "@openclaw/normalization-core/record-coerce";
import { Value } from "typebox/value";
import {
  ErrorCodes,
  errorShape,
  type SessionMemberIdentity,
} from "../../../packages/gateway-protocol/src/index.js";
import { SessionExecutionDispatchParamsSchema } from "../../../packages/gateway-protocol/src/schema/sessions-viewer-presence.js";
import {
  privateRoomExecutionForRun,
  withPrivateRoomExecution,
  type PrivateRoomExecution,
} from "../../agents/private-room-execution.js";
import { resolveSessionWorkStartError } from "../../config/sessions/lifecycle.js";
import { privateRoomPolicyForEntry } from "../../config/sessions/private-room-policy.js";
import {
  findTranscriptEvent,
  readActiveTranscriptEntryAnchor,
} from "../../config/sessions/session-accessor.js";
import {
  readTranscriptEventId,
  readTranscriptEventMessage,
} from "../../config/sessions/session-accessor.sqlite-read.js";
import { isSessionMember } from "../../config/sessions/session-sharing-store.js";
import {
  getAgentRunContext,
  getAgentRunContextOwnerStatus,
} from "../../infra/agent-run-registry.js";
import { prepareAgentRequestPreflight } from "../agent-turn/agent-request-preflight.js";
import { createAgentTurnService } from "../agent-turn/agent-turn-service.js";
import { captureAgentTurnPrincipal } from "../agent-turn/principal.js";
import type { AgentTurnIo } from "../agent-turn/types.js";
import type { ChatAbortControllerEntry } from "../chat-abort.js";
import { registerPrivateRoomExecution } from "../private-room-executions.js";
import {
  authorizeSessionAgentRun,
  authorizeSessionSharingTarget,
  resolveSessionSharingTarget,
} from "../session-sharing-policy.js";
import type { GatewayRequestHandler } from "./types.js";

/** Only this transition admits inference; the referenced contribution is already committed. */
export const dispatchSessionExecution: GatewayRequestHandler = async (options) => {
  const { params, client, context, respond } = options;
  if (!Value.Check(SessionExecutionDispatchParamsSchema, params)) {
    respond(
      false,
      undefined,
      errorShape(ErrorCodes.INVALID_REQUEST, "Invalid private room execution request"),
    );
    return;
  }
  const request = structuredClone(params);
  let close = () => {};
  let accepted = false;
  try {
    const runtime = client?.internal?.agentRuntimeIdentity;
    const profileId = client?.authenticatedUserProfile?.profileId;
    const identity: SessionMemberIdentity | undefined = runtime
      ? { type: "agent", id: runtime.agentId }
      : profileId && !client?.internal?.syntheticClient
        ? { type: "profile", id: profileId }
        : undefined;
    if (!identity) {
      throw new Error("Execution requires an authenticated profile or live agent run");
    }
    const parent = runtime ? privateRoomExecutionForRun(runtime.operationalRunInstance) : undefined;
    if (runtime && (!parent || context.validateAgentRuntimeApprovalAuthority?.(runtime) !== true)) {
      throw new Error("Delegation requires an authenticated private root execution");
    }
    const hopCount = parent ? parent.hopCount + 1 : 0;
    if (hopCount > 3 || (request.hopCount !== undefined && request.hopCount !== hopCount)) {
      throw new Error(
        "Private room delegation exceeds or disagrees with the authenticated hop count",
      );
    }
    const target = resolveSessionSharingTarget({
      cfg: context.getRuntimeConfig(),
      sessionKey: request.sessionKey,
    });
    const policy = target ? privateRoomPolicyForEntry(target.entry) : undefined;
    if (!target || !policy) {
      throw new Error("Private room is unavailable");
    }
    const scope = {
      agentId: target.agentId,
      sessionKey: target.storeKey,
      sessionId: request.expectedSessionId,
      storePath: target.storePath,
    };
    if (
      parent &&
      (parent.sessionId !== scope.sessionId || parent.sessionKey !== target.canonicalKey)
    ) {
      throw new Error("Private room delegation cannot cross rooms");
    }
    let closed = false;
    let runOwner: ChatAbortControllerEntry | undefined;
    let started = false;
    const runId = createHash("sha256")
      .update(
        JSON.stringify([
          "room-execution-v1",
          scope.sessionId,
          identity,
          request.idempotencyKey,
          request.inputMessageId,
        ]),
      )
      .digest("hex");
    const assertCurrent = () => {
      if (closed || client?.invalidated || client?.connectionSignal?.aborted) {
        throw new Error("Private room execution authority was revoked");
      }
      if (
        !runtime &&
        (client?.authenticatedUserProfile?.profileId !== profileId ||
          client?.internal?.syntheticClient)
      ) {
        throw new Error("Private room principal changed");
      }
      if (!accepted) {
        options.sessionMutationCommitGuard?.();
      }
      options.sessionMutationAuthorization?.assertCurrent();
      parent?.assertCurrent();
      if (runtime && context.validateAgentRuntimeApprovalAuthority?.(runtime) !== true) {
        throw new Error("Parent run authority was revoked");
      }
      const cfg = context.getRuntimeConfig();
      const current = resolveSessionSharingTarget({ cfg, sessionKey: request.sessionKey });
      if (
        !current ||
        current.storeKey !== scope.sessionKey ||
        current.storePath !== scope.storePath ||
        resolveSessionWorkStartError(scope.sessionKey, current.entry, {
          expectedSessionId: scope.sessionId,
        }) ||
        JSON.stringify(privateRoomPolicyForEntry(current.entry)) !== JSON.stringify(policy) ||
        authorizeSessionSharingTarget({ cfg, client, target: current }) ||
        !isSessionMember(scope, identity) ||
        !isSessionMember(scope, { type: "agent", id: target.agentId })
      ) {
        throw new Error("Private room membership or session authority was revoked");
      }
      if (!readActiveTranscriptEntryAnchor({ ...scope, entryId: request.inputMessageId })) {
        throw new Error("Execution input is not an active committed room message");
      }
      if (
        runOwner &&
        (runOwner.controller.signal.aborted ||
          context.chatAbortControllers.get(runId) !== runOwner ||
          runOwner.sessionId !== scope.sessionId ||
          runOwner.sessionKey !== target.canonicalKey)
      ) {
        throw new Error("Private room run authority was revoked");
      }
      if (started) {
        const run = getAgentRunContext(runId);
        const authority = runOwner?.agentRunDelegatedAuthority;
        // Inspect the claim without re-entering its assertSourceCurrent callback.
        if (
          !run ||
          run.agentId !== target.agentId ||
          run.sessionKey !== target.canonicalKey ||
          !authority ||
          run.delegatedAuthority !== authority ||
          authority.operationalRunInstance !== runOwner?.operationalRunInstance ||
          getAgentRunContextOwnerStatus(runId, authority.claimId, authority.lifecycleGeneration) ===
            undefined
        ) {
          throw new Error("Private room run is no longer active");
        }
      }
    };
    assertCurrent();
    // ponytail: scan the canonical transcript for one input; add an indexed exact event reader if measured latency requires it.
    const contribution = await findTranscriptEvent(
      scope,
      (event) => readTranscriptEventId(event) === request.inputMessageId,
    );
    assertCurrent();
    const message = readTranscriptEventMessage(contribution?.event);
    const sender = asOptionalRecord(asOptionalRecord(message?.["__openclaw"])?.senderIdentity);
    if (
      !message ||
      message.role !== "user" ||
      sender?.type !== "profile" ||
      !Array.isArray(message.content)
    ) {
      throw new Error("Execution input must be a committed human message");
    }
    const text = message.content
      .flatMap((part: unknown) =>
        part &&
        typeof part === "object" &&
        "type" in part &&
        part.type === "text" &&
        "text" in part &&
        typeof part.text === "string"
          ? [part.text]
          : [],
      )
      .join("\n");
    if (!text.trim()) {
      throw new Error("Execution input has no text");
    }
    let release = () => {};
    let heartbeat: ReturnType<typeof setInterval> | undefined;
    const presenceTransport = `private-run:${runId}`;
    close = () => {
      if (closed) {
        return;
      }
      closed = true;
      if (heartbeat) {
        clearInterval(heartbeat);
      }
      context.nativeRoomPresence?.disconnect(presenceTransport, true);
      release();
    };
    const execution: PrivateRoomExecution = Object.freeze({
      agentId: target.agentId,
      rootExecutionId: parent?.rootExecutionId ?? runId,
      runId,
      hopCount,
      sessionKey: target.canonicalKey,
      sessionId: scope.sessionId,
      inputMessageId: request.inputMessageId,
      assertCurrent,
      close,
    });
    release = registerPrivateRoomExecution({
      assertCurrent,
      abort: () => {
        runOwner?.controller.abort(new Error("Private room membership revoked"));
        close();
      },
    });
    const agentRequest = {
      message: text,
      sessionKey: target.canonicalKey,
      agentId: target.agentId,
      expectedExistingSessionId: scope.sessionId,
      idempotencyKey: runId,
      suppressPromptPersistence: true,
      deliver: false,
      disableMessageTool: true,
      cwd: policy.sessionRoot,
      workspaceDir: policy.sessionRoot,
    };
    const io: AgentTurnIo = {
      emitStartOwner: (_runId, entry) => {
        runOwner = entry;
        assertCurrent();
      },
      emitExecutionStarted: () => {
        started = true;
        assertCurrent();
        const beat = () => {
          try {
            assertCurrent();
            context.nativeRoomPresence?.update(
              presenceTransport,
              target.canonicalKey,
              {
                actor: { type: "agent", id: target.agentId },
                isAuthorized: () => {
                  try {
                    assertCurrent();
                    return true;
                  } catch {
                    return false;
                  }
                },
              },
              {
                heartbeat: true,
                visibility: "visible",
                recentInput: "recent",
                viewingIntent: "viewing",
              },
            );
          } catch {
            runOwner?.controller.abort(new Error("Private room authority revoked"));
            close();
          }
        };
        beat();
        if (!closed) {
          heartbeat = setInterval(beat, 30_000);
          heartbeat.unref();
        }
      },
      emitAcceptance: ([ok, payload, error], meta) => {
        if (ok) {
          assertCurrent();
          accepted = true;
        } else {
          close();
        }
        respond(
          ok,
          ok
            ? {
                ...(payload && typeof payload === "object" ? payload : {}),
                persisted: true,
                executionAccepted: true,
                inputMessageId: request.inputMessageId,
                rootExecutionId: execution.rootExecutionId,
                hopCount,
              }
            : payload,
          error,
          meta,
        );
        if (meta?.cached) {
          close();
        }
      },
      emitFinal: () => {
        close();
      },
    };
    await withPrivateRoomExecution(execution, async () => {
      const accessError = authorizeSessionAgentRun({
        cfg: context.getRuntimeConfig(),
        client,
        target,
      });
      if (accessError) {
        throw new Error(accessError.message);
      }
      const principal = captureAgentTurnPrincipal(client);
      const preflight = prepareAgentRequestPreflight({
        request: agentRequest,
        context,
        client: principal,
        io,
      });
      if (!preflight) {
        return;
      }
      await createAgentTurnService(
        { context, isWebchatConnect: options.isWebchatConnect },
        assertCurrent,
      ).startTurn({ preflight, principal, io, assertAdmissionCurrent: assertCurrent });
    });
    if (!accepted) {
      close();
    }
  } catch (error) {
    close();
    if (!accepted) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          error instanceof Error ? error.message : "Private room execution unavailable",
        ),
      );
    }
  }
};
