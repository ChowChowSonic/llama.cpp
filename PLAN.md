# PLAN: MoE inference improvements for this llama.cpp fork

Derived from three papers: FreeToken (freetokens_paper.pdf), vLLM/PagedAttention
(vllm.pdf), SGLang (sglang.pdf), and the state of this fork. Ordered into tiers
by expected payoff per unit of work. Each item lists the mechanism, the
llama.cpp/Vulkan integration point, and how to validate it.

FreeToken drives the MoE-specific tiers (0-4). The vLLM and SGLang papers
contribute the KV-state and scheduling tier (5): they are complementary to the
MoE work because every prefill token avoided is also ~one expert pool streamed
over PCIe.

Baseline: this fork already has a working shared-LRU expert cache for
host-resident MoE weights (used with `-cmoe` / `-ncmoe` on Vulkan) and
tool-call-aware state checkpoints in the server.

## Backend priority

Vulkan is the primary and only priority backend for this fork. All
implementation, optimization, profiling, and testing work targets Vulkan
(the dev box is an Intel Arc Pro B70 / BMG G31; no CUDA on this machine).
CUDA, Metal, ROCm, etc. are explicitly out of scope; do not spend effort
adapting changes to them or worrying about breaking them beyond what the
compiler enforces. The CPU backend matters only in two roles: host-resident
expert weights (source of truth for the cache) and as the CPU-execution
branch of the q* policy; it is also the fallback config for correctness
baselines.

---

## Tier 0 - already implemented (verify, then build on)

1. Semantic checkpoint anchors at user + tool boundaries (Phase 1)
   - `common/chat.h`: `is_anchor_start()` / `last_anchor_pos()`
   - `tools/server/server-context.cpp`: batch breaks and checkpoint decisions
     use anchors instead of user-only spans
   - Validate: multi-turn tool-loop request logs "restored context checkpoint"
     on the second turn; TTFT of turn N does not grow with edited history

2. Shared LRU expert cache, host weights as source of truth (Phase 2)
   - `ggml/src/ggml-backend-expert-cache.{h,cpp}`
   - One device slot pool per tensor, uniform slot count across layers,
     expert IDs remapped host-side to slot IDs, `MUL_MAT_ID` src[0] rewired
     to the pool view
   - `--expert-cache-size N` (MiB), `n_expert_cache_bytes` in context params
   - Validate: greedy output identical with cache 0 vs cache > 0

---

## Tier 1 - decode path (the paper's biggest wins)

3. q* bandwidth-adaptive miss serving (paper sec. 3.2)
   - Today every cache miss is staged into VRAM before compute (serial PCIe,
     then GPU). The paper splits the m unique misses of a layer into
     q* = m * B_P / B_H fills + (m - q*) experts executed in place on CPU,
     concurrently, then merges the two partial sums exactly.
   - B_P: measured host->device expert transfer bandwidth (schedule a probe
     copy at startup, like the existing device scan timing)
   - B_H: measured CPU MoE kernel bandwidth (probe a MUL_MAT_ID on CPU at
     startup)
   - Integration: in `ggml_backend_expert_cache_prepare`, instead of returning
     false or staging everything when misses exceed slots, partition the miss
     set. GPU branch: fills + hits. CPU branch: a second MUL_MAT_ID whose
     src[0] is the host weight stack with the CPU-served expert IDs; result
     merged with an ADD on the GPU (exact sum, gate weights unchanged).
   - This removes the current hard fallback: "too many distinct experts for
     the pool" no longer falls back to full per-ubatch copies, it shifts
     work to the CPU instead.
   - Validate: decode tok/s vs expert-cache-size sweep; bit-exact greedy
     parity still holds (merge order must match reference MUL_MAT_ID)

4. Router locality metrics and tuning knobs
   - The paper's premise: adjacent decode tokens reuse most of the same
     experts. Expose cache stats (hits/misses/fallbacks already in the
     struct) via `-dgs`/log or a `/metrics` endpoint on the server.
   - Try better replacement than plain LRU: LFU-with-aging, or seed slots
     from GGUF per-expert activation stats (`*.weight` frequency tensors
     that GGUFs carry) as a warm start on first use of a layer.

5. Fused/grouped expert staging copy
   - Current miss path issues one `ggml_backend_tensor_set_async` per expert
     per tensor. Group misses that are contiguous in the host stack and
     coalesce into fewer transfers; on Vulkan this means fewer
     `vkCmdCopyBuffer` calls. A single device-side index list (paper sec. 4.1)
     is the end state, but host-side coalescing captures most of the gain.

---

## Tier 2 - prefill path (TTFT)

6. Full-layer double-buffered expert streaming during prefill (sec. 3.1)
   - Prefill routes through almost every expert, so the per-miss cache is
     the wrong granularity; the whole layer's experts must cross PCIe anyway.
   - Reserve 2 full-layer buffers from the same slot pool; while the GPU
     computes layer l, stream layer l+1 on a second Vulkan transfer queue.
     Transfer can start before routing is known because the full layer is
     loaded. Fall back to on-demand (current path) if the pool cannot spare
     two full layers.
   - Prefill survivors seed decode residency (already true: one pool).
   - Integration: the copy scheduling in
     `ggml_backend_sched_compute_splits` is serialized per split; needs a
     look-ahead of one layer plus a second copy stream on Vulkan
     (`vulkan_queue` / async upload path already exists for host buffers).
   - Validate: prefill tok/s at 4k/8k/16k with overlap on/off; paper
     measures 19-26% at those lengths.

7. Widen semantic anchors to thinking segments (sec. 3.1)
   - Harnesses strip thinking blocks from older assistant turns; anchors
     should also sit at assistant-thinking span boundaries so those edits
     also resolve to an anchor. `common_chat_msg_span` already distinguishes
     content kinds; extend `is_anchor_start` accordingly.

8. Recurrent-state checkpoints for hybrid models (sec. 2.1 / 3.1)
   - For GDN/KDA hybrid models (Qwen3.5/3.6-Next family, this fork's target
     model), state checkpoints are expensive; placement at anchors matters
     more than count. The existing `llama_state_seq_*` + `n_rs_seq` rollback
     covers the mechanics; what is missing is the anchor-aware restore:
     pick the deepest checkpoint whose position survives the edited prompt,
     including the recurrent state, not just KV.
   - Validate on a hybrid-attention model with history truncation.

---

## Tier 3 - elasticity and startup (sec. 3.3, 4.2)

9. Runtime cache resize at scheduler safe points
   - Currently the pool is sized once (first evals), disabled on failure.
     Allow rebuild for a revised budget: evict everything, free slot buffers,
     reallocate, keep host weights (source of truth), invalidate graphs.
   - This also enables reacting to VRAM pressure from other apps and to KV
     growth over long sessions (the KV-vs-expert split becomes tunable).

10. Dynamic KV-cache vs expert-cache split
    - After Tier 3.9 exists, expose a policy: on context growth, shrink slot
      pool to hand VRAM to KV (or the reverse), instead of the static split
      chosen from `--expert-cache-size` alone.

11. Fast bootstrap
    - Load expert pool directly into final host layout then pin/mmap; avoid
      GPU warmup by serving the first request with a cold cache (already the
      behavior; make sure the reserve/warmup path in llama-context skips the
      MoE warmup graph when the cache is on, or make it tolerant of cold
      cache).
    - Longer term: a pre-packed layout where each (layer, expert) is one
      contiguous row across ffn_up/gate/down (paper's FTW / "expert banks").
      For llama.cpp this means either converting GGUFs to reorder ffn_*_exps
      into per-expert-contiguous shards, or a loader that memcpys into
      banked host buffers. Removes per-tensor discovery and makes every
      expert copy a single flat range.

12. Pure-CPU MoE fallback when pinned/DMA path is unavailable
    - llama.cpp `-cmoe` already covers this; make the cache fall back to it
      cleanly on hosts where buffer pinning fails (Windows, some drivers).

---

## Tier 4 - device-side control, graph capture (sec. 4.1) - deferred

13. Keep routing-dependent cache decisions on the device
    - Today ids are read back to the host to remap expert -> slot IDs. That
      sync is invisible next to CPU-copy latency but fatal under graph
      capture. The paper runs dedup, residency classification, q* sizing,
      victim selection and ID rewrite inside one kernel per layer.
    - In llama.cpp/Vulkan this needs a residency table in a device buffer
      and a small compute shader; only worth starting once the Vulkan
      graph-capture work can actually consume it.

14. Graph-resident CPU branch
    - CPU experts submitted through a captured graph node (paper uses a
      host-function node). CUDA-specific; Vulkan equivalent would need
      host-side timeline semaphores. Park until Tier 4.13.

---

## Tier 5 - KV state management and scheduling (vllm.pdf, sglang.pdf)

15. Copy-on-write prompt sharing for parallel sampling (vllm sec. 4.4)
    - When the server decodes n > 1 samples from one prompt (multi-slot
      parallel sampling, fork/join fan-out), the prompt KV is currently
      computed or duplicated per sequence. Map all sequences' logical blocks
      for the shared prompt to one set of physical blocks with a refcount;
      copy-on-write only the block being written.
    - Integration: unified KV cache (`src/llama-kv-cache*.cpp`) already
      decouples rows from slots; add per-block refcounts and a COW hook on
      write. Same idea applies to beam-search-style rescheduling, where
      surviving candidates keep shared blocks instead of memcpys.
    - Validate: memory headroom and decode tok/s for n-generate=4 with a
      long shared prompt; outputs unchanged.

16. Radix-tree prefix cache across requests (sglang sec. 3)
    - Keep the KV of finished requests in a radix tree keyed by token
      sequence instead of discarding it; new prompts match the deepest
      prefix and skip that prefill entirely. LRU eviction of leaves only
      (so shared ancestors survive until they become leaves), refcounts to
      protect blocks used by running requests, tree kept on host with
      negligible overhead.
    - This generalizes this fork's per-slot checkpoints (Tier 0.1) to
      cross-request reuse for full attention, and composes with semantic
      anchors: anchors decide where recurrent state may be restored, the
      radix tree decides what full-attention KV survives. FreeToken's
      runtime already assumes a radix prefix tree for KV plus anchor-attached
      recurrent checkpoints; this item supplies the missing first half.
    - Integration: server slot lifecycle in
      `tools/server/server-context.cpp` currently drops prompt KV on
      completion; add an insert-on-finish, match-on-arrival pass over the
      unified KV cache. Start with exact system-prompt matching (item 19)
      before general splits.
    - Validate: multi-turn chat and few-shot batch workloads - TTFT drops
      proportional to matched prefix length; memory bounded by tree LRU.

17. Cache-aware scheduling (sglang sec. 3)
    - Order the waiting queue by longest matched prefix first (equivalent to
      a DFS traversal of the request radix tree, which is provably optimal
      offline). Avoids thrashing when many related requests interleave.
    - In this fork that means the server's slot selection / prompt ordering
      consults the prefix cache from item 16 before choosing what to
      prefill next. Guard against starvation with an aging bound.
    - Depends on 16. Validate: aggregate TTFT on interleaved multi-session
      replay vs FIFO ordering.

18. Preemption with host-swap or recompute (vllm sec. 4.5)
    - Under KV pressure (long multi-turn sessions + expert cache already
      squeezed VRAM), evict whole-sequence KV of the newest requests rather
      than fragmenting running ones. Two recovery modes:
      swap to host RAM (bounded: swapped blocks <= GPU KV blocks) or
      recompute - and recompute is cheap here precisely because the anchor
      checkpoints from Tier 0 exist.
    - llama.cpp can already keep some layers' KV on the CPU
      (`-ot kv_store`) but has no request-level preempt/resume. The policy
      choice (swap vs recompute) mirrors the paper's bandwidth argument:
      on edge PCIe links recompute usually wins.
    - Validate: oversubscribe slots/contexts deliberately; no request
      should fail, worst-case latency stays bounded.

19. Pre-registered shared prefixes (vllm sec. 4.4)
    - Reserve physical KV blocks for known long system prompts / few-shot
      templates once at startup; incoming requests map their first blocks
      onto the cached blocks (last block COW), prefill runs only on the task
      suffix. For this fork: the agent harness system contexts (24.5k-token
      floors in the FreeToken traces) make this a large repeated saving.
    - Natural special case of 16; do it first as the stepping stone.

20. Compressed-FSM multi-token grammar jumps (sglang sec. 4)
    - Constrained decoding (JSON schemas, fixed reply formats used by agent
      harnesses) walks deterministic FSM runs one token per forward pass
    - Compress maximal single-transition paths of the grammar FSM into one
      jump: when only one token sequence can follow, append it without a
      model pass. llama.cpp's grammar engine already does partial jumps for
      some deterministic transitions; the CFSM formulation (compress
      adjacent single-edge states at grammar-compile time, detect at
      sampling time) is the general version and covers constant runs like
      `{"summary": "` spanning several tokens.
    - Integration: the grammar/constraint engine used by the server's
      structured output path; keep it exact, no sampling change.
    - Validate: JSON-mode decode steps vs wall time on fixed schemas;
      output distribution unchanged.

---

## Cross-cutting work items

- Budget sizing helper: print per-layer expert bytes and suggested
  `--expert-cache-size` values at load (like the `-sm`/fit output) so users
  can pick 10%/25% of pool sizes that the paper evaluates.
- `llama-bench` support for `--expert-cache-size` sweeps (add a field the
  way `no_cpu_moe` is threaded).
- Test matrix: greedy parity (cache 0 vs N), slot exhaustion (budget forces
  fallback path), prefill-heavy vs decode-heavy prompts, multi-turn
  checkpoint restore. Reuse existing tests/snapshots infra; no new test
  binaries.
- CPU backend correctness: the cache is a no-op on CPU-only scheds (no
  host->device split), confirm nothing regresses with cache flag set.

## Suggested order

Tier 1 item 3 (q* split) is the largest single decode win and reuses all the
residency bookkeeping that already exists. Then item 6 (pipelined prefill)
for TTFT, then 9-11 for usability, and treat Tier 4 as blocked on
graph-capture support in the Vulkan backend.

In the KV tier, item 19 (pre-registered prefixes) is a small change with
immediate payoff for agent workloads and the natural first step toward 16
(radix cache), which then unblocks 17 (cache-aware scheduling). Item 20
(grammar jumps) is independent of everything else and can be done any time.
Item 15 matters only when multi-sequence sampling from a shared prompt is
actually used; keep it behind 16, whose block-sharing machinery subsumes it.

---

## Notes for future sessions

Read this before touching anything below.

### Ground rules for this repo
- This is a private fork for personal experimentation. The upstream
  llama.cpp contribution rules (issue-first discussion, PR scoping, etc.)
  do not apply; use judgment and keep changes minimal anyway.
- Do not interact with Git or GitHub on the user's behalf: no commits,
  no pushes, no PRs, no `gh` writes, unless explicitly asked each time.
- Code style: concise comments only where the code cannot say it itself,
  ASCII only (no unicode arrows/dashes), follow existing patterns in the
  files being edited. Write code first, comments after, only where needed.

### Environment
- GPU: Intel Arc Pro B70 (BMG G31) via Vulkan; also the iGPU. No NVIDIA.
- Build: `cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release`
  then `cmake --build build -j24 --target llama-cli llama-server`.
  Build dir `build/` is already configured this way.
- Test model (MoE, ~80 GB, 3 shards, pass shard 00001):
  `~/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/824f539b2710e5a9e47af4952cf6578cf5ee8932/UD-Q2_K_XL/`
  Use `-cmoe -ngl 99` to force host-resident experts, `--temp 0 -s 42`
  for determinism, `--no-jinja` for raw completion (`-no-cnv` no longer
  exists in this CLI). 27B dense and mmproj files also in the HF cache.

### State of the fork (as of Tier 0)
- Working tree has uncommitted work (do not revert): the expert cache
  (`ggml/src/ggml-backend-expert-cache.{h,cpp}` + hooks in
  `ggml/src/ggml-backend.cpp`), Phase 1 anchors (`common/chat.h`,
  `tools/server/server-context.cpp`), param plumbing (`include/llama.h`
  `n_expert_cache_bytes`, `src/llama-cparams.h`, `src/llama-context.cpp`
  around sched creation ~line 604, `common/common.h`
  `expert_cache_size` MiB, `common/common.cpp` to_llama mapping,
  `common/arg.cpp` `--expert-cache-size` / `LLAMA_ARG_EXPERT_CACHE_SIZE`).
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` carries the user's own large
  pre-existing changes. Never reformat or regenerate that file; read it
  carefully before editing and keep edits surgical.
- Build is clean. Cache path smoke-tested by user and now verified
  automatically (see Phase 1 progress below).
- Cache enable line at runtime: look for `expert cache: N slots/layer` in
  stderr from GGML_LOG_INFO; warnings on that prefix indicate fallback to
  the legacy per-ubatch copy path. NOTE: llama-cli/llama-server suppress
  GGML INFO logs unless run with `-v`; without it the enable line is
  invisible even when the cache is on.

### Phase 1 progress (verified)
- CRASH FIXED (real bug, not just the user's accidental second instance):
  `ggml_backend_expert_cache_entry_get` was called for every split input in
  `ggml_backend_sched_compute_splits`, and it registered any tensor (e.g.
  the `ffn_moe_topk` ids tensor, ne [10,2,1] nb2=4096) as an expert-stack
  group. The bogus group (n_expert = ids->ne[2] = 1) forced global
  n_slots = 1 and its pool memset OOB'd (`tensor write out of bounds` at
  ggml-backend.cpp:411 during the load-time `common_context_can_seq_rm`
  probe). Fix: new lookup-only
  `ggml_backend_expert_cache_entry_find_w()` used at the call site;
  registration now happens inside the block only for genuine
  MUL_MAT_ID host-weight inputs (`node->src[0] == input_cpy`). Files:
  `ggml/src/ggml-backend-expert-cache.{h,cpp}`, `ggml/src/ggml-backend.cpp`.
- KEY ENVIRONMENT GOTCHA: by default MUL_MAT_ID is placed on the CPU
  backend (host weights), so the expert stacks never cross a split and the
  cache never engages. Vulkan only claims host-weight ops when
  `ggml_vk_get_op_batch_size(op) >= GGML_OP_OFFLOAD_MIN_BATCH` (default
  32, ggml-vulkan.cpp ~19178/19338), so without the override the cache is
  prefill-only. All cache testing MUST set `GGML_OP_OFFLOAD_MIN_BATCH=1`
  to get decode-path engagement.
- Greedy parity PASSED: same prompt/seed/temp 0, `-cmoe -ngl 99
  --no-jinja`, GGML_OP_OFFLOAD_MIN_BATCH=1, `--expert-cache-size 0` vs
  `8192`: generated text byte-identical (only the t/s line differs).
- Cache enable confirmed with `-v`: `expert cache: 95 slots/layer,
  144 tensors, 8154.8 MiB` (48 layers x 3 stacks, 512 experts/layer),
  no warnings. Server on port 8090 also shows it.
- Throughput data points (tiny prompt, 64 tokens): CPU-MoE default
  placement decoded ~13 t/s; Vulkan MoE with legacy per-ubatch copies
  (cache 0, env=1) ~3.5 t/s; cache 8192 with env=1 ~3.5 t/s on the first
  short request (cold cache, no warmup budget). Decode win needs a longer
  generation to measure warm hit rate - open question for Phase 2.
- Model facts: qwen4exp GGUF (Qwen3.8-Flash-Next UD-Q2_K_XL) has 3 shards,
  experts live in shards 2-3; 48 MoE layers, 512 routed experts/layer,
  gate/up [2560,640,512] + down [640,2560,512], separate (not fused)
  gate/up tensors, shared expert per layer.
- Server multi-turn checkpoint-restore test PASSED (cache 8192 active,
  `-cmoe -ngl 99 -c 8192 --temp 0 -s 42 -v`, env above, requests via
  `curl -d @file.json`):
  req1 804-token prompt -> 3 checkpoints created (pos 46 anchor, 287, 799
  via the 4+n_ubatch / 4 offsets). req2 extends history -> restored
  checkpoint 799, cached 800/837, 9 s wall. req3 edits a word ~token 350
  (after a checkpoint was erased by the min-spacing rule) -> log shows
  the reverse-order checkpoint scan and `restored context checkpoint
  (pos_min = 46, ...)`, 761/808 tokens re-prefilled at 43 t/s instead of
  a full reset, decode 3.2 t/s, `graphs reused = 19`.
  Note: checkpoints are erased when closer than `checkpoint_min_step`
  (here spacing 8192 -> only anchor/last survive), so deep restores can
  degenerate to the first anchor. Relevant to item 8 gap (c).
- Phase 1 gate result: Tier-0 stack verified end-to-end. Two follow-ups
  carried into Phase 2: warm hit-rate throughput win is unmeasured, and
  the checkpoint min-spacing/selection behavior above.
- Execution agreement: phase-by-phase with user gates. Approved choices:
  item 3 validated by practical equivalence (byte-exact only when the q*
  split path is inactive; identical greedy tokens + small logit tolerance
  when active); item 19 implemented as persistent template state blob
  (full `llama_state_seq` prefill-once, load per matching request), NOT
  KV row sharing. Agreed order: verify Tier 0 -> item 3 -> item 8 -> 19
  -> 6 -> 16.

### Key code findings for Phase 2/3 (from exploration)
- Item 3 integration point: the fallback branch at
  `ggml_backend.cpp` (the `if (!cached)` block after
  `ggml_backend_expert_cache_prepare` returns false, ~line 1750).
  Bandwidth probes belong in `ggml_backend_expert_cache_size`. CPU branch
  cannot insert graph nodes post-allocation; plan is: read activation x
  from device (small during decode), run MUL_MAT_ID via a micro-graph on
  the CPU backend against the host stack, merge partial on device.
- Item 8: `state_seq` save/load ALREADY covers GDN S-states, conv rows
  and PLE conv history (`llama-memory-recurrent.cpp` state_write_data);
  server checkpoints use PARTIAL_ONLY which for hybrid_idx saves only the
  recurrent part (KV + indexer KV re-derived after restore). Real gaps:
  (a) `LLM_ARCH_QWEN4EXP` missing from `llm_arch_supports_rs_rollback`
  (src/llama-arch.cpp ~1099) so n_rs_seq = 0; (b) restore/seq_rm ordering
  on hybrid memory; (c) deepest-surviving-anchor checkpoint selection.
- Item 19: template blob = a never-evicted checkpoint with pos_min==0;
  server checkpoint store is `common_prompt_checkpoint`
  (common/common.cpp update_tgt/dft via llama_state_seq_get_data_ext).
  Server restore/search logic at server-context.cpp ~3322-3358.

### Expert cache design invariants (violating any of these breaks it)
- Host expert weights are the source of truth; device slot pools are pure
  cache, never written back.
- Slot count is uniform across all layers: `budget / sum(per-expert bytes
  over all groups)`, capped at n_expert. Groups are keyed by (backend,
  layer index parsed from `blk.N` in the tensor name, else name hash).
- Groups are discovered during the first eval(s) on the legacy copy path;
  pools are built at the top of the next `ggml_backend_sched_compute_splits`
  call. Tensors that first appear after sizing log "late tensor" and stay
  on the copy path forever.
- When the cache serves a node it mutates the graph: `src[0]` -> pool,
  `src[2]` -> slot-remapped ids view. On graph reuse the node may still
  point at the cache; detection is via `entry->pool`/`ids_orig` backpointers
  (`ggml_backend_expert_cache_entry_find`). Any change to graph reuse,
  sched reset, or copy handling in `ggml-backend.cpp` must preserve the
  restore-on-fallback path and `ggml_backend_expert_cache_invalidate` on
  sched reset.
- Constraints: single split backend copy (warns and disables under
  pipeline parallelism), ids must be contiguous I32, per-eval distinct
  experts must fit in n_slots or the layer falls back to full copies
  (until Tier 1 item 3 replaces that with the q* split).
- Pools are zero-initialized so MMQ tile over-reads never see NaNs; if a
  future change evicts/reuses slots with garbage, keep that property or
  fix the kernels.

### Verification recipes
- Parity: same prompt/seed/`--temp 0`, compare full outputs with
  `--expert-cache-size 0` vs e.g. 8192 while running `-cmoe`; must be
  byte-identical.
- TTFT/anchor test: multi-turn request through llama-server where the
  second turn edits history after a tool call; second-turn prefill should
  resume from a checkpoint (server debug logs `restored context checkpoint`
  / `main/do_checkpoint`).
- Throughput: `llama-bench` or server decode tok/s sweeps; note that with
  `-cmoe` the expert copies dominate, so cache effects show up mainly in
  decode tok/s as slot hit rate rises.
