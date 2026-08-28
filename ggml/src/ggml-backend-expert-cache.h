#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Persistent device-side LRU cache for MoE expert weights. The host-resident expert weights
// stay the source of truth. On each MUL_MAT_ID eval, missing experts are staged into fixed
// device slots and reused across evals by rewriting the node inputs (src[0] -> expert pool,
// src[2] -> slot-remapped ids). Ref: FreeToken paper, sec. 3.2 semantic-aware expert caching

struct ggml_backend_expert_cache_entry;

struct ggml_backend_expert_cache_group {
    ggml_backend_t backend = nullptr;
    int64_t key = 0; // layer index parsed from the tensor name, or a name hash

    int32_t n_expert = 0;     // shared by all tensors in the group
    int32_t n_slots  = 0;     // 0 = not sized yet

    std::vector<int32_t>  slot_of;     // [n_expert] expert -> slot, -1 = not resident
    std::vector<int32_t>  slot_expert; // [n_slots]  slot -> expert, -1 = free
    std::vector<uint64_t> last_use;    // [n_expert]
    std::vector<uint8_t>  used_mark;   // [n_expert] scratch, routed experts of current eval

    int32_t n_resident = 0;
    size_t  expert_bytes = 0; // total bytes per expert across the group tensors

    std::vector<ggml_backend_expert_cache_entry *> entries;
};

struct ggml_backend_expert_cache_entry {
    ggml_tensor * w = nullptr; // host expert stack, MUL_MAT_ID src[0]
    ggml_backend_t backend = nullptr;
    ggml_backend_expert_cache_group * group = nullptr;

    size_t expert_size = 0; // bytes per expert in this tensor

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * pool = nullptr; // device view, ne[2] = group->n_slots, nullptr = legacy

    // ids tensors recorded when the cache first claims a node. reused graphs keep the node
    // rewired to the cache view, so raw ids are read from ids_orig. ids_cpy is what the
    // node read originally, restored when falling back to the copy path. both become
    // dangling after a scheduler reset, see ggml_backend_expert_cache_invalidate
    ggml_tensor * ids_cpy = nullptr;
    ggml_tensor * ids_orig = nullptr;
    ggml_backend_t ids_orig_backend = nullptr;
};

struct ggml_backend_expert_cache {
    size_t budget = 0;
    size_t used   = 0;

    bool sized    = false; // slot tables and pools allocated
    bool disabled = false;

    uint64_t clock = 0;

    ggml_context * ctx = nullptr; // owns pool tensors and ids views

    std::vector<ggml_backend_expert_cache_group *> groups;
    std::vector<ggml_backend_expert_cache_entry *> entries;

    struct ids_view {
        ggml_backend_expert_cache_group * group;
        ggml_backend_t backend;
        int64_t ne0;
        int64_t ne1;
        ggml_backend_buffer_t buf;
        ggml_tensor * t;
    };
    std::vector<ids_view> views;

    std::vector<int32_t> ids_host; // staging for remapped ids

    // stats
    uint64_t n_hit = 0;
    uint64_t n_miss = 0;
    uint64_t n_fallback = 0;
};

ggml_backend_expert_cache * ggml_backend_expert_cache_create(size_t budget);
void ggml_backend_expert_cache_free(ggml_backend_expert_cache * cache);

// allocate the slot tables and device pools once the first full eval has observed all groups
// no-op when already sized or disabled
void ggml_backend_expert_cache_size(ggml_backend_expert_cache * cache);

// drop recorded node input tensors after a scheduler reset, the graph is about to be rebuilt
void ggml_backend_expert_cache_invalidate(ggml_backend_expert_cache * cache);

// entry serving (backend, w), or NULL when the cache cannot be used. created lazily
ggml_backend_expert_cache_entry * ggml_backend_expert_cache_entry_get(
        ggml_backend_expert_cache * cache, ggml_backend_t backend, ggml_tensor * w);

// existing entry for (backend, w), without registering new groups or entries
ggml_backend_expert_cache_entry * ggml_backend_expert_cache_entry_find_w(
        ggml_backend_expert_cache * cache, ggml_backend_t backend, ggml_tensor * w);

// entry owning the pool tensor t, or NULL. used to recognize reused graphs where the node
// src still points at the pool
ggml_backend_expert_cache_entry * ggml_backend_expert_cache_entry_find(
        ggml_backend_expert_cache * cache, ggml_backend_t backend, ggml_tensor * t);

// update residency and stage missing experts into slots. ids holds the raw routed expert
// ids, [ne0, ne1] contiguous. returns false when the eval cannot be served (too many
// distinct experts, pool allocation pending) and the caller must use the regular copy path
bool ggml_backend_expert_cache_prepare(
        ggml_backend_expert_cache * cache, ggml_backend_expert_cache_entry * entry,
        ggml_backend_t backend, const int32_t * ids, int64_t ne0, int64_t ne1);

// remap ids to slot ids, upload them to a per-(group, shape) device tensor and return it
// NULL on allocation failure, caller must then use the regular copy path
ggml_tensor * ggml_backend_expert_cache_ids_view(
        ggml_backend_expert_cache * cache, ggml_backend_expert_cache_entry * entry,
        ggml_backend_t backend, const int32_t * ids, int64_t ne0, int64_t ne1);
