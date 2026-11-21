/**
 * Thin C shim over the real Aker library (github.com/sjoon-oh/aker, built
 * unmodified as libaker.so), so a Python harness can drive it with one call
 * per operation. Aker's caching logic is not reimplemented here; this file
 * only follows the authors' integration protocol (apps/pgvector/pgvector.patch):
 *
 *   lookup   akerGetCacheEntry; on an approximate hit, allocate a dummy entry
 *            for the query and link it to the representative
 *            (akerLinkCacheEntry) so a repeat of the same query is an exact hit
 *   miss     the caller searches the backend, then insert: one vector slot
 *            per neighbor (with its squared L2 distance to the query, the unit
 *            Aker's thresholds use), akerCreateCacheEntry, akerInsertCacheEntry
 *   writes   akerInsertWriteLogEntry (fast path) + akerProcessWriteLogEntries
 *            (slow path); deletes: akerMarkVectorDeleted
 *
 * Each call reports its own time (steady clock, seconds) for the Aker work.
 */

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ak_anns_cache_c_wrapper.h"
#include "ak_logger.hh"

namespace
{
    using clk = std::chrono::steady_clock;

    struct Shim
    {
        anns_cache_c_wrapper_t* cache = nullptr;
        std::uint32_t dim = 0;
        std::size_t bytes = 0;
        std::size_t topk = 0;
        std::size_t delta = 0;
        std::string status;
    };

    bool copyTransform(void* src, size_t src_size, size_t dim, void* dst, uint8_t* aux)
    {
        (void)aux;
        if (src == nullptr || dst == nullptr || src_size < dim * sizeof(float))
            return false;
        std::memcpy(dst, src, dim * sizeof(float));
        return true;
    }

    float l2Bytes(uint8_t* a, uint8_t* b, size_t dim)
    {
        return akerL2Distance(reinterpret_cast<float*>(a), reinterpret_cast<float*>(b), dim);
    }

    /* Write-log results are only read as vector ids by our harness: no payload conversion. */
    void noResultConversion(uint64_t, uint8_t*, size_t, uint64_t, uint64_t) {}

    double seconds(clk::time_point a, clk::time_point b)
    {
        return std::chrono::duration<double>(b - a).count();
    }
}

extern "C"
{
    /* risk, alpha_tighten, alpha_loosen: the [tuning] section; pool = vector slots. */
    void* shim_create(std::uint32_t dim, std::uint64_t pool, std::uint32_t topk, std::uint32_t delta,
                      float risk, float alpha_tighten, float alpha_loosen, const char* config_path)
    {
        Shim* s = new Shim();
        /* Aker never initializes its logger here, so Boost.Log's default sink would print
         * every debug line (several per request) on the hot path; keep warnings only. */
        aker::Logger::setMinLevel(aker::Logger::Level::k_warning);
        anns_cache_parameter_c_t p{};
        akerImportAnnsCacheConfig(const_cast<char*>(config_path), &p);
#ifdef RECAL_AKER_FILTER
        /* Built against the vendored library (third_party/aker), which exposes the query filter's HNSW
         * parameters. Defaults are the paper's: m 4, efConstruction 8, efSearch 16 (the unmodified library
         * hard-codes efSearch 8 and efConstruction 16). */
        auto env_int = [](const char* name, int fallback) {
            const char* v = std::getenv(name);
            return v ? std::atoi(v) : fallback;
        };
        p.filter.hnsw_m = env_int("AKER_FILTER_M", 4);
        p.filter.ef_construction = env_int("AKER_FILTER_EFC", 8);
        p.filter.ef_search = env_int("AKER_FILTER_EFS", 16);
#endif
        p.vector_format.dimension = dim;
        p.vector_format.vector_in_bytes = static_cast<std::size_t>(dim) * sizeof(float);
        p.capacity.pool_size = pool;
        p.capacity.in_topk = topk;
        p.capacity.top_delta = delta;
        p.tuning.risk_thresh = risk;
        p.tuning.alpha_tighten = alpha_tighten;
        p.tuning.alpha_loosen = alpha_loosen;
        /* Proximity-mode builds (AKER_ENABLE_PROXIMITY_MODE=ON) use one fixed global threshold
         * (squared L2, like every Aker distance); read from AKER_GLOBAL_THRESH if set. Standard
         * mode ignores it. */
        if (const char* g = std::getenv("AKER_GLOBAL_THRESH"))
            p.tuning.global_thresh = static_cast<float>(std::atof(g));
        p.distance_metric = 0;
        s->dim = dim;
        s->bytes = p.vector_format.vector_in_bytes;
        s->topk = topk;
        s->delta = delta;
        s->cache = akerCreateAnnsCache(p);
        if (s->cache == nullptr)
        {
            delete s;
            return nullptr;
        }
        return s;
    }

    void shim_destroy(void* h)
    {
        Shim* s = static_cast<Shim*>(h);
        if (s == nullptr)
            return;
        akerDestroyAnnsCache(s->cache);
        delete s;
    }

    /* Returns 0 miss, 1 exact hit, 2 approximate hit, 3 invalid entry, 4 linked hit
     * (repeat of a query served approximately before). On a hit,
     * out_ids[0 .. topk) are the cached neighbor ids; t[0] = lookup seconds,
     * t[1] = alias-link seconds (approximate hits). */
    int shim_lookup(void* h, std::uint64_t qid, const float* q, std::int64_t* out_ids, double* t)
    {
        Shim* s = static_cast<Shim*>(h);
        const auto t0 = clk::now();
        char* slot = akerCreateVectorSlot(qid, s->bytes, reinterpret_cast<char*>(const_cast<float*>(q)),
                                          0, 0, 0.0f);
        char* view = akerCreateVectorView(slot, s->dim, s->bytes, copyTransform);
        bool similar = false;
        bool invalid = false;
        char* entry = akerGetCacheEntry(s->cache, view, &similar, &invalid, l2Bytes);
        const auto t1 = clk::now();
        t[0] = seconds(t0, t1);
        t[1] = 0.0;
        int kind = 0;
        if (entry != nullptr && !invalid)
        {
            kind = similar ? 2 : 1;
            /* A repeat of a query that was earlier served approximately (it was linked
             * to a representative): Aker counts it as a sim-hit; the answer is the
             * representative's. */
            if (!similar && akerGetQueryVectorIdFromCacheEntry(entry) != qid)
                kind = 4;
            for (std::size_t i = 0; i < s->topk; i++)
                out_ids[i] = static_cast<std::int64_t>(
                    akerGetVectorIdFromVectorSlot(akerGetResultVectorSlotAt(entry, static_cast<int>(i))));
            if (similar)
            {
                const auto t2 = clk::now();
                const std::uint64_t found_id = akerGetQueryVectorIdFromCacheEntry(entry);
                char* alias = akerCreateCacheEntry(s->cache, slot, s->topk + s->delta, nullptr);
                (void)akerLinkCacheEntry(s->cache, alias, found_id);
                t[1] = seconds(t2, clk::now());
            }
        }
        else if (entry != nullptr)
        {
            kind = 3;
        }
        if (entry != nullptr)
            akerDestroyCacheEntry(entry);
        akerDestroyVectorView(view);
        akerDestroyVectorSlot(slot);
        return kind;
    }

    /* Insert the entry for a query after a backend search: n = topk + delta
     * neighbors (ids, and their vectors n x dim, row-major). Returns 1 if inserted;
     * *t = seconds. */
    int shim_insert(void* h, std::uint64_t qid, const float* q, std::uint32_t n, const std::int64_t* nid,
                    const float* nvec, double* t)
    {
        Shim* s = static_cast<Shim*>(h);
        const auto t0 = clk::now();
        float* qm = const_cast<float*>(q);
        char* qslot = akerCreateVectorSlot(qid, s->bytes, reinterpret_cast<char*>(qm), 0, 0, 0.0f);
        char* qview = akerCreateVectorView(qslot, s->dim, s->bytes, copyTransform);
        std::vector<char*> slots;
        slots.reserve(n);
        for (std::uint32_t j = 0; j < n; j++)
        {
            float* v = const_cast<float*>(nvec) + static_cast<std::size_t>(j) * s->dim;
            const float dist = akerL2Distance(qm, v, s->dim);
            slots.push_back(akerCreateVectorSlot(static_cast<std::uint64_t>(nid[j]), s->bytes,
                                                 reinterpret_cast<char*>(v), 0, 0, dist));
        }
        char* entry = akerCreateCacheEntry(s->cache, qslot, n, slots.data());
        const bool inserted = akerInsertCacheEntry(s->cache, qid, entry, qview);
        if (!inserted)
            akerDestroyCacheEntry(entry);
        for (char* sl : slots)
            akerDestroyVectorSlot(sl);
        akerDestroyVectorView(qview);
        akerDestroyVectorSlot(qslot);
        *t = seconds(t0, clk::now());
        return inserted ? 1 : 0;
    }

    /* A vector inserted into the database: write-log fast path. */
    double shim_write(void* h, std::uint64_t id, const float* v)
    {
        Shim* s = static_cast<Shim*>(h);
        const auto t0 = clk::now();
        char* slot = akerCreateVectorSlot(id, s->bytes, reinterpret_cast<char*>(const_cast<float*>(v)),
                                          0, 0, 0.0f);
        char* view = akerCreateVectorView(slot, s->dim, s->bytes, copyTransform);
        akerInsertWriteLogEntry(s->cache, view, l2Bytes, noResultConversion);
        akerDestroyVectorView(view);
        akerDestroyVectorSlot(slot);
        return seconds(t0, clk::now());
    }

    /* Write-log slow path (the authors call it after each request). */
    double shim_process(void* h)
    {
        Shim* s = static_cast<Shim*>(h);
        const auto t0 = clk::now();
        akerProcessWriteLogEntries(s->cache, l2Bytes, noResultConversion);
        return seconds(t0, clk::now());
    }

    /* A vector deleted from the database. */
    double shim_delete(void* h, std::uint64_t id)
    {
        Shim* s = static_cast<Shim*>(h);
        const auto t0 = clk::now();
        akerMarkVectorDeleted(s->cache, id);
        return seconds(t0, clk::now());
    }

    const char* shim_status(void* h)
    {
        Shim* s = static_cast<Shim*>(h);
        const char* text = akerGetCacheStatusText(s->cache);
        s->status = text == nullptr ? "" : text;
        return s->status.c_str();
    }
}
