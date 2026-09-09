# Task 3 report — private execution and native presence

## Independent review follow-up: literal committed input

Authenticated private dispatch now bypasses command/reset interpretation at the shared `runAgentResetPhase` boundary before matching or lifecycle effects. The existing private execution context is the authority; no client-controlled mode or alternate inference service was added. The embedded prompt submission also uses the existing `expandPromptTemplates: false` option in that context, suppressing extension commands and skill/template expansion. Normal chat retains both behaviors. Existing agent-turn admission, billing/hold, and inference remain unchanged.

Caller trace: ordinary `agentRunHandler`, internal agent-turn facade, and private `sessions.execution.dispatch` all enter the same agent-turn service, whose only reset interpreter is `runAgentResetPhase`. Only authenticated private dispatch establishes the private context. Dynamic reset scope detection applies to the ordinary `agent` RPC only. Embedded submission reaches `AgentSession.prompt`, where the existing expansion option also gates extension commands; embedded resource loading already disables ambient extensions/templates/skills. Source regressions cover literal `/reset`, `/new`, reset-with-follow-up, no reset lifecycle callbacks/ack, unchanged ordinary handling, and a real session that sends `/reset` to a fake model without invoking a registered command. No local validation or formatter was executed.

## Independent review follow-up: durable archive privacy

Window provenance now carries a monotonic `memory_restricted` classification (`1` restricted, `0` durably ordinary, `NULL` unknown). The existing archive persistence boundary copies it into the canonical archive row inside the deletion transaction, before lifecycle reclamation removes the window and logical room. Retained inventory and archive discovery use this stored classification; deleting the room cannot make its text eligible for global indexing/dreaming. No transcript store or archive format was added.

The existing additive-schema migration path adds nullable columns without a schema-version bump. Migration backfills only exact, valid surviving session identities and their matching archives. Previously deleted or otherwise unclassifiable archives stay unknown. Read-only old-schema readers use unknown without migrating. Parent-approved compatibility ruling: unknown registered archives and filename-only legacy artifacts are excluded, including files that merely resemble a live room's archive. Names and live metadata do not establish archive privacy. Durably ordinary registered archives remain eligible. This intentionally reduces historical memory coverage until trustworthy classification exists; it does not delete archive files.

Source regressions exercise canonical archive creation followed by complete room/window deletion and database reopen (restricted excluded, ordinary retained), idempotent additive migration with unknown/deleted identities, and filename-only artifact exclusion. Prior corpus assertions that admitted unclassified files were updated for the new fail-closed contract. CI must execute the migration, archive lifecycle/corpus, and command-boundary regressions; none were run locally.

Only the required deterministic write-mode generators ran for this correction: `node --import ./scripts/tsx.mjs scripts/generate-kysely-types.mts` and `node --import ./scripts/tsx.mjs scripts/generate-sqlite-session-schema-baseline.ts --write`. They updated the committed agent database types and schema fingerprint; the state database types were rewritten identically, and the baseline also emitted its normal ignored `.artifacts/sqlite-session-transcript-schema-baseline.sql` inspection artifact. No test, lint, typecheck, build, check, formatting-check, formatter, or autoreview was executed.

## Independent review follow-up: completion-hook privacy

The independent review identified a global-memory leak through `agent_end`: LanceDB's completion consumer extracts and persists user text. The shared `createHookRunner().runAgentEnd` dispatcher now suppresses all completion-hook consumers whenever the existing private execution context is present, including completion after revocation/closure. Ordinary executions retain their existing hook dispatch. This is not a LanceDB-specific exception.

Caller trace: embedded `attempt-finalize` and CLI `cli-run-transcript`, plus Copilot and Codex provider wrappers, reach `runAgentEndSideEffects` / `awaitAgentEndSideEffects`, then the single `executeAgentHarnessAgentEndHook` caller of `runAgentEnd`. Both awaited and fire-and-forget variants preserve the async-local private execution context; private v1 independently rejects non-OpenClaw harnesses. Direct hook-runner consumers meet the same root fence. The separate internal session `agent_end` lifecycle event is not the plugin `agent_end` dispatcher and was not modified.

Added `src/plugins/hooks.private-room.test.ts` using the real hook dispatcher and two registered consumers. Source cases cover successful/failed private completions, an awaited boundary, revoked completion, and unchanged ordinary dispatch afterward. No tests, lint, typecheck, build, checks, formatter, or autoreview were executed for this correction.

## Delivery

- Worktree: `/Users/prasiddhaparthsarthy/Desktop/isol8.nosync/.worktrees/openclaw-room-isolation-presence`
- Branch: `codex/openclaw-room-isolation-presence`
- Original stacked base: `9882de4e5d5d4db76cdd2bb168af323e188a83fe`
- Feature commit: `3a60b79e03` — `feat: isolate private room execution and add native presence` (75 files, 2,481 additions, 56 deletions).
- This report and the local ruling ledger are committed separately after the feature commit. No push, rebase, merge, or cherry-pick was performed. Parent owns transplanting the feature onto final Task 2 head `e5ae8e207919a4b4221a953a26838b14af13befd` and opening/updating the stacked PR.

## Implemented boundaries

- Immutable exact-session policy validation/default capabilities in `src/config/sessions/private-room-policy.ts`, creation policy selection, and sandbox runtime-status/workspace selection. Invalid, missing, mismatched, or unsupported policy fails closed.
- `src/agents/sandbox/private-room.ts` uses the existing `SandboxContext` and pinned filesystem bridge inside the founder container. There is no room container, founder-workspace mount, or host-tool fallback. Only the internal pinned file helper transport can run a command; the exposed backend refuses exec and browser capabilities. Private roots and runtime identifiers are session-specific.
- `src/agents/agent-tools.ts` and the existing embedded run/compaction construction seams retain only explicitly allowed file tools. Later external MCP/LSP/client-tool, tool-search/code-mode, global skill/bootstrap, plugin prompt-enrichment, and custom context-engine paths cannot expand private v1 authority. Provider-hosted search continues to consume the same sandbox tool policy.
- `src/gateway/server-methods/sessions-files.ts` resolves only the room root, rejects repository fallback, and uses native mutation/read authorization rechecks. File tools retain the bridge's traversal/symlink/pinned-mutation protections and check live execution before and after work.
- Memory tool construction, execution, query, and recall visibility deny restricted requesters. Active Memory and prompt hooks cannot provide automatic global recall. Existing memory corpus discovery excludes restricted active/retained transcript identities and matching archive artifacts from global indexing/dreaming; no second room corpus was added.
- `sessions.execution.dispatch` accepts a committed human message ID, not new model input or client-authored ancestry. It derives the root/run identity on the host, rejects excessive/disagreeing hop counts before admission, and uses existing agent-turn admission/inference/billing. Dispatch denial leaves the canonical input intact. Private v1 runs the room-owning agent; it does not add a generic agent mesh or target-selection API.
- `src/agents/private-room-execution.ts`, the existing admission/abort owner, transcript write seams, and `sessions.message.append` enforce the exact session instance and live operational run. Model/hook assistant results are host-stamped with the executing agent identity. Membership changes synchronously revalidate active owners and safely abort revoked room work; stale result append is rejected independently.
- Native presence v1 lives in the existing Gateway lifecycle/subscription organization. Profile intents require authenticated active WebSockets and exact-session ACLs; agent presence comes from live admitted execution. Server-owned connection IDs, epoch timestamps and connection-local sequence, simultaneous connections, reconnects, 30-second heartbeats, 90-second lease expiry, five-minute input activity, one-second typing throttle, 2.5-second typing expiry, final-only last seen, and immediate membership revocation are implemented.
- Protocol reconciliation preserves native complete/incomplete/failed inventory status, rejects poisoned ordering evidence, aggregates separate connection IDs, treats invalid/missing evidence as unknown, and advances server time only with caller-supplied monotonic elapsed time. Viewing declarations remain independent from transport subscriptions.

## Main changed areas

| Area | Files / existing seams |
| --- | --- |
| Policy and execution | `src/config/sessions/private-room-policy.ts`, `types.ts`, `src/agents/private-room-execution.ts`, `src/gateway/private-room-executions.ts`, `server-methods/sessions-execution.ts`, existing agent-turn/command/transcript paths |
| Files and tools | `src/agents/sandbox/{private-room,context,runtime-status}.ts`, `agent-tools.ts`, embedded-run tool/bootstrap/skill/compaction paths, `src/gateway/server-methods/sessions-files.ts` |
| Memory | `extensions/memory-core/src/{private-room,memory-tool-contract,memory-search-tool-query,session-search-visibility,tools.shared}.ts`, plugin hooks, Active Memory, `src/context-engine/registry.ts`, `packages/memory-host-sdk/src/host/session-transcript-corpus.ts` |
| Presence | `src/gateway/native-room-presence.ts`, `native-room-presence-authority.ts`, `server-methods/sessions-presence.ts`, Gateway lifecycle/context/WebSocket/sharing/viewing/typing/subscription wiring |
| Protocol | New native-presence and dispatch schemas, validators/registry exports, native-presence projection, core descriptors, event registration, generated Swift/Kotlin artifacts |
| Documentation | `docs/gateway/protocol/presence.md`, `docs/gateway/protocol/rpc-methods.md`, this report and `progress.md` |

## Test source and execution record

Focused source-only cases cover private root/path/symlink isolation and unsupported policy, exact run binding/revocation/hop ceiling, canonical append surviving admission rejection, memory-only and cross-session recall denial, global indexing exclusion, no private skill environment/context, no private external bundle runtime acquisition, run-derived append authority, presence leases/throttle/expiry/reconnect/last-seen/ACL failure, closed human RPCs, and snapshot/evidence ordering.

Added or extended test files:

- `src/agents/private-room-execution.test.ts`
- `src/agents/sandbox/private-room.test.ts`
- `src/agents/embedded-agent-runner/run/attempt-startup.test.ts`
- `src/agents/embedded-agent-runner/run/attempt-bundle-tools.test.ts`
- `src/gateway/server-methods/sessions-execution.test.ts`
- `src/gateway/server-methods/sessions-message-append.test.ts`
- `extensions/memory-core/src/private-room.test.ts`
- `packages/memory-host-sdk/src/host/session-files.test.ts`
- `src/gateway/native-room-presence.test.ts`
- `src/gateway/server-methods/sessions-presence.test.ts`
- `packages/gateway-protocol/src/native-presence-projection.test.ts`

**None were executed locally.** No test, lint, typecheck, build, check, formatting-check, formatter, or autoreview command ran. Commit hooks were disabled to preserve that restriction. Source self-review was manual: traced construction through late bundle/catalog/compaction stages, admission and operational authority ownership, low-level transcript writes, current ACLs, and presence lifecycle/evidence projection.

Only authorized deterministic source generators ran successfully:

1. `node --import ./scripts/tsx.mjs scripts/protocol-gen-swift.ts` — updated committed `apps/shared/OpenClawKit/Sources/OpenClawProtocol/GatewayModels.swift`.
2. `node --import ./scripts/tsx.mjs scripts/protocol-gen-kotlin.ts` — updated committed `apps/android/app/src/main/java/ai/openclaw/app/gateway/GatewayProtocol.kt`; the constants output was rewritten identically and has no diff.

`protocol.schema.json` is an untracked build artifact and was not generated. Generator success is not a test/typecheck/build result.

## Risks and CI/deployment gaps

- The implementation is unverified by executed tests, compiler, lint, formatter, or CI. GitHub CI must exercise the focused suites, protocol registry/generated-source consistency, source formatting and typechecking, plus existing Gateway/embedded-run/memory regressions after the stack transplant.
- The approved backend is **capability isolation, not an OS process sandbox**. Its pinned filesystem helpers require the founder container's Linux shell/Python/helper environment. An ECS/Fargate image smoke test must prove file reads/writes/patches, path escape denial, and revocation; shell/browser capabilities must remain rejected until a separately reviewed backend can satisfy them.
- Full live-provider dispatch/result and quota/abort/WebSocket-reconnect integration were not exercised. Browser UI adapters must explicitly send the two opening operations (viewing and subscriptions), heartbeat, and render native evidence without browser-wall-clock inference.
- v1 executes the room-owning member agent. The default file-only tool surface intentionally has no cross-session/delegation tool; the authenticated same-room dispatch lineage is available at the native runtime boundary without introducing generic mesh behavior.
- The committed-input lookup scans the canonical transcript; measure before adding an indexed exact-event lookup. Global corpus discovery now reads retained policy metadata even when retained transcript export was not requested so deletion cannot erase the privacy fence; observe that cost on large stores.
- Presence is bounded ephemeral state (4,096 connection/lease slots with bounded per-connection rooms/tombstones). Capacity or uncertain ACL inventory is not represented as a complete empty/offline snapshot.
- Existing immutable private policies are not migrated. Older policies without reviewed capabilities remain unable to execute/file-access until product-owned migration decisions are made; no implicit upgrade or historical global-memory purge was attempted.
