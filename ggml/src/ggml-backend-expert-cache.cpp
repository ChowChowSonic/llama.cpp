#include "ggml-backend-expert-cache.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>

static int64_t ggml_expert_cache_group_key(const char * name) {
    const char * p = strstr(name, "blk.");
    if (p != nullptr) {
        char * end = nullptr;
        const long v = strtol(p + 4, &end, 10);
        if (end != p + 4) {
            return v;
        }
    }

    int64_t h = (int64_t) 1469598103934665603ULL;
    for (const char * c = name; *c; ++c) {
        h = (h ^ (uint8_t) *c) * (int64_t) 1099511628211ULL;
    }
    return h | ((int64_t) 1 << 40);
}

ggml_backend_expert_cache * ggml_backend_expert_cache_create(size_t budget) {
    ggml_backend_expert_cache * cache = new ggml_backend_expert_cache;

    cache->budget = budget;

    const ggml_init_params params = {
        /* .mem_size =   */ 4 * 1024 * 1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc =   */ true,
    };

    cache->ctx = ggml_init(params);
    if (cache->ctx == NULL) {
        delete cache;
        return NULL;
    }

    return cache;
}

void ggml_backend_expert_cache_free(ggml_backend_expert_cache * cache) {
    if (cache == NULL) {
        return;
    }

    // the pool and ids view tensors live in cache->ctx, free the buffers first
    for (auto * entry : cache->entries) {
        if (entry->buf != NULL) {
            ggml_backend_buffer_free(entry->buf);
        }
        delete entry;
    }

    for (auto & view : cache->views) {
        ggml_backend_buffer_free(view.buf);
    }

    for (auto * group : cache->groups) {
        delete group;
    }

    ggml_free(cache->ctx);

    delete cache;
}

void ggml_backend_expert_cache_size(ggml_backend_expert_cache * cache) {
    if (cache == NULL || cache->sized || cache->disabled) {
        return;
    }

    if (cache->groups.empty()) {
        return; // no offloaded MoE weights seen yet
    }

    size_t total = 0;
    int32_t min_expert = INT32_MAX;

    for (auto * group : cache->groups) {
        total += group->expert_bytes;
        min_expert = std::min(min_expert, group->n_expert);
    }

    if (total == 0) {
        cache->disabled = true;
        return;
    }

    // uniform slot count so that all layers share the budget evenly
    const int64_t n_slots = std::min<int64_t>(cache->budget / total, min_expert);

    if (n_slots < 1) {
        cache->disabled = true;
        GGML_LOG_WARN("expert cache: budget %.1f MiB too small, disabled\n", cache->budget / 1024.0 / 1024.0);
        return;
    }

    for (auto * group : cache->groups) {
        group->n_slots = n_slots;
        group->slot_of.assign(group->n_expert, -1);
        group->slot_expert.assign(n_slots, -1);
        group->last_use.assign(group->n_expert, 0);
        group->used_mark.assign(group->n_expert, 0);
        group->n_resident = 0;
    }

    // all layers must be cached or none, partial residency breaks the shared slot mapping
    bool ok = true;

    for (auto * entry : cache->entries) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(entry->backend);
        const size_t size = entry->expert_size * n_slots;

        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size);
        if (buf == NULL) {
            ok = false;
            break;
        }

        ggml_tensor * t = ggml_dup_tensor(cache->ctx, entry->w);
        t->ne[2] = n_slots;

        if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ok = false;
            break;
        }

        // zero so that MMQ tile reads past the last slot never see NaNs
        ggml_backend_tensor_memset(t, 0, 0, size);

        entry->buf  = buf;
        entry->pool = t;
        cache->used += size;
    }

    if (!ok) {
        cache->disabled = true;
        GGML_LOG_WARN("expert cache: pool allocation failed, disabled\n");
        return;
    }

    cache->sized = true;

    GGML_LOG_INFO("expert cache: %d slots/layer, %zu tensors, %.1f MiB\n",
            (int) n_slots, cache->entries.size(), cache->used / 1024.0 / 1024.0);
}

void ggml_backend_expert_cache_invalidate(ggml_backend_expert_cache * cache) {
    if (cache == NULL) {
        return;
    }

    for (auto * entry : cache->entries) {
        entry->ids_cpy = nullptr;
        entry->ids_orig = nullptr;
        entry->ids_orig_backend = nullptr;
    }
}

ggml_backend_expert_cache_entry * ggml_backend_expert_cache_entry_get(
        ggml_backend_expert_cache * cache, ggml_backend_t backend, ggml_tensor * w) {
    if (cache == NULL || cache->disabled) {
        return NULL;
    }

    for (auto * entry : cache->entries) {
        if (entry->w == w && entry->backend == backend) {
            return entry->pool != NULL ? entry : NULL;
        }
    }

    if (cache->sized) {
        // groups are discovered during the first eval, new groups cannot be sized
        GGML_LOG_WARN("expert cache: late tensor %s, staying on the copy path\n", w->name);
        return NULL;
    }

    const int64_t key = ggml_expert_cache_group_key(w->name);

    ggml_backend_expert_cache_group * group = NULL;
    for (auto * g : cache->groups) {
        if (g->key == key && g->backend == backend) {
            group = g;
            break;
        }
    }

    if (group == NULL) {
        group = new ggml_backend_expert_cache_group;
        group->key = key;
        group->backend = backend;
        group->n_expert = w->ne[2];
        cache->groups.push_back(group);
    } else if (group->n_expert != w->ne[2]) {
        GGML_LOG_WARN("expert cache: mismatched expert count for %s, staying on the copy path\n", w->name);
        return NULL;
    }

    ggml_backend_expert_cache_entry * entry = new ggml_backend_expert_cache_entry;

    entry->w = w;
    entry->backend = backend;
    entry->group = group;
    entry->expert_size = w->nb[2];

    group->entries.push_back(entry);
    group->expert_bytes += entry->expert_size;

    cache->entries.push_back(entry);

    return NULL; // the pool does not exist until the cache is sized
}

ggml_backend_expert_cache_entry * ggml_backend_expert_cache_entry_find(
        ggml_backend_expert_cache * cache, ggml_backend_t backend, ggml_tensor * t) {
    if (cache == NULL || t == NULL) {
        return NULL;
    }

    for (auto * entry : cache->entries) {
        if (entry->pool == t && entry->backend == backend) {
            return entry;
        }
    }

    return NULL;
}

bool ggml_backend_expert_cache_prepare(
        ggml_backend_expert_cache * cache, ggml_backend_expert_cache_entry * entry,
        ggml_backend_t backend, const int32_t * ids, int64_t ne0, int64_t ne1) {
    ggml_backend_expert_cache_group * group = entry->group;

    const int32_t n_expert = group->n_expert;
    const int32_t n_slots  = group->n_slots;

    // distinct routed experts
    static thread_local std::vector<int32_t> used;

    used.clear();
    std::memset(group->used_mark.data(), 0, n_expert);

    const int64_t n_ids = ne0 * ne1;

    for (int64_t i = 0; i < n_ids; i++) {
        const int32_t id = ids[i];
        assert(id >= 0 && id < n_expert);
        if (!group->used_mark[id]) {
            group->used_mark[id] = 1;
            used.push_back(id);
        }
    }

    if ((int32_t) used.size() > n_slots) {
        cache->n_fallback++;
        return false;
    }

    for (const int32_t id : used) {
        if (group->slot_of[id] >= 0) {
            group->last_use[id] = ++cache->clock;
            cache->n_hit++;
        }
    }

    for (const int32_t id : used) {
        if (group->slot_of[id] >= 0) {
            continue;
        }

        int32_t victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int32_t s = 0; s < n_slots; s++) {
            const int32_t cur = group->slot_expert[s];
            if (cur < 0) {
                victim = s;
                break;
            }
            if (group->used_mark[cur]) {
                continue;
            }
            if (group->last_use[cur] < oldest) {
                oldest = group->last_use[cur];
                victim = s;
            }
        }

        if (victim < 0) {
            cache->n_fallback++;
            return false;
        }

        const int32_t evicted = group->slot_expert[victim];
        if (evicted >= 0) {
            group->slot_of[evicted] = -1;
        } else {
            group->n_resident++;
        }

        ggml_backend_tensor_set_async(backend, entry->pool,
                (const char *) entry->w->data + (size_t) id * entry->expert_size,
                (size_t) victim * entry->expert_size, entry->expert_size);

        group->slot_of[id]     = victim;
        group->slot_expert[victim] = id;
        group->last_use[id]    = ++cache->clock;

        cache->n_miss++;
    }

    return true;
}

ggml_tensor * ggml_backend_expert_cache_ids_view(
        ggml_backend_expert_cache * cache, ggml_backend_expert_cache_entry * entry,
        ggml_backend_t backend, const int32_t * ids, int64_t ne0, int64_t ne1) {
    ggml_backend_expert_cache_group * group = entry->group;

    ggml_backend_expert_cache::ids_view * found = NULL;
    for (auto & view : cache->views) {
        if (view.group == group && view.backend == backend && view.ne0 == ne0 && view.ne1 == ne1) {
            found = &view;
            break;
        }
    }

    if (found == NULL) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, ne0 * ne1 * sizeof(int32_t));
        if (buf == NULL) {
            return NULL;
        }

        ggml_tensor * t = ggml_new_tensor_2d(cache->ctx, GGML_TYPE_I32, ne0, ne1);

        if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            return NULL;
        }

        ggml_backend_expert_cache::ids_view view;
        view.group  = group;
        view.backend = backend;
        view.ne0 = ne0;
        view.ne1 = ne1;
        view.buf = buf;
        view.t = t;

        cache->views.push_back(view);
        found = &cache->views.back();
    }

    cache->ids_host.resize(ne0 * ne1);
    for (int64_t i = 0; i < ne0 * ne1; i++) {
        cache->ids_host[i] = group->slot_of[ids[i]];
    }

    ggml_backend_tensor_set(found->t, cache->ids_host.data(), 0, ne0 * ne1 * sizeof(int32_t));

    return found->t;
}
