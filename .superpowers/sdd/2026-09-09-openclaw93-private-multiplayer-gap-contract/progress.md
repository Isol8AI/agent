# SDD ledger — plan: /Users/prasiddhaparthsarthy/Desktop/isol8.nosync/docs/superpowers/plans/2026-09-09-openclaw93-private-multiplayer-gap-contract.md

## Task 3 preflight interface scan

| Producer / consumer                             | Interface                                                                | Ruling                                                                                                                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task 1 -> Task 3                                | Restricted visibility, typed membership, atomic room policy, revocation  | Consume the canonical ACL only; participants, profile presentation, and operator write do not grant membership.                                            |
| Task 2 -> Task 3                                | Authenticated non-inference append and run-derived agent result identity | Execution dispatch is a separate state machine; it may reference a committed input message but cannot rewrite or roll it back.                             |
| Task 3 isolation -> Task 3 files/memory/tools   | Exact session-owned private execution policy                             | One immutable room policy and exact session root must feed every downstream capability check; no per-profile or global-workspace fallback.                 |
| Task 3 dispatch -> Task 3 revocation/result     | Authenticated root execution id, hop count, live membership              | Reject hop count greater than three before work; recheck membership and exact session instance for dispatch and result append; abort safely on revocation. |
| Task 3 presence producer -> Task 5/UI consumers | Native v1 server-timestamped leases, snapshots, events, completeness     | Preserve complete/incomplete/failed inventory; viewing intent and subscription stay independent; no message-recency or client-wall-clock inference.        |

Ruling: implement the plan's full trust-boundary and timing contract, but reuse existing sandbox, session-policy, subscription, clock, and participant primitives before adding state. No second transcript, room container, or global mesh.
Ruling: the task may use scoped commits for isolation/dispatch and presence inside one independently reviewable stacked PR; this does not combine fork PR boundaries.
Ruling: user forbids local test/lint/typecheck/build/check execution. Tests remain code-as-evidence and GitHub CI is the execution gate.

Task 3: active from stacked base `9882de4e5d` (`codex/openclaw-room-isolation-presence`).

## Task 3 implementation rulings and handoff

- Parent-approved sandbox ruling: use the existing `SandboxContext` / pinned filesystem bridge seam inside the founder container. ECS/Fargate cannot provision a nested Docker/Podman sandbox, and the contract forbids one container per room. The private backend is explicitly file-capability-only; every exposed operation is bounded by the immutable exact-session root and live execution authority. It is not an OS process sandbox. Shell/browser or other unsupported future capabilities are rejected, with no founder-workspace or host-tool fallback.
- Private policy now flows through creation, runtime status, workspace selection, coding tools, the embedded tool/catalog and compaction paths, file RPCs, prompt/skill/context-engine boundaries, and global-memory visibility/corpus discovery. The default has only read/write/edit/apply_patch file tools and exact-room file/dispatch RPC capabilities. Old immutable policies are not upgraded in place.
- Dispatch references the already committed human message; it alone enters existing agent-turn admission and billing. Host-owned execution ancestry, exact operational-run binding, per-use ACL/lifecycle/input checks, host-stamped results, and revocation aborts remain separate from transcript persistence.
- Native presence v1 uses the existing Gateway lifecycle and subscription paths, bounded ephemeral leases and tombstones, authenticated profile intents, runtime-owned agent signals, server epoch plus monotonic elapsed time, and complete/incomplete/failed snapshot reconciliation. Viewing declarations and subscriptions remain independent.
- Local validation remains forbidden and was not executed. Focused tests were added as source only. No test/lint/typecheck/build/check/formatting-check/autoreview command ran. Only the authorized deterministic Swift/Kotlin protocol source generators ran, to update their committed artifacts; no formatter or JSON build-artifact generator ran.
- Implementation stays on `9882de4e5d5d4db76cdd2bb168af323e188a83fe`; no rebase/merge/cherry-pick/push occurred. Parent owns transplanting onto the final lower Task 2 stack and the GitHub CI gate.

## Task 3 independent-review corrections

- Authenticated committed room input is literal model input, never a Gateway reset or embedded slash command. Reuse the private execution context and existing `expandPromptTemplates: false`; keep ordinary chat and inference/admission/billing unchanged.
- Archive privacy must survive complete logical-room/window deletion. Carry a sticky nullable privacy classification through the existing window and canonical archive rows, and copy it before deletion inside the same transaction.
- Parent ruling: unknown archive privacy is never public. Exclude legacy registered rows without recoverable classification and filename-only artifacts from global indexing/dreaming; preserve durably ordinary registered archives. Do not infer privacy from filenames or later live metadata. The historical-memory coverage reduction is intentional; no archive files are removed.
- No local validation ran. Added source regressions and ran only the required deterministic Kysely type and SQLite schema-baseline write generators; see the report for artifacts and remaining CI coverage.
