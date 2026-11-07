// Replays a request stream through Aker (standard or proximity mode, chosen at
// compile time). The backend is read from a precomputed answer table.
//
// Files (little endian):
//   --base      float32 rows, int32 (n, d) header, then n*d floats
//   --requests  float32 rows, int32 (m, d) header: the vector of every request
//   --stream    int64[m]: universe id of every request
//   --answers   int64[m * k]: backend top-k ids per request (nearest first)
//   --dists     float32[m * k]: Euclidean distances of those answers
// Output (--out-dir): outcome.u8 (0 miss, 1 equality hit: served by an entry stored under
// this exact vector id, 2 approximate hit via the filter, 3 linked repeat served by another
// query's entry, counted as approximate), result_ids.i64 (m * k, hit lists or backend answers),
// seen.u8 (request id seen before), status.log, summary.txt.
//
// Distances handed to Aker are squared L2: slot distances are the squared
// table distances and hits are tested with the library's own squared L2.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ak_anns_cache_c_wrapper.h"

namespace
{
    struct Mapped
    {
        const char* data{nullptr};
        size_t bytes{0};
    };

    Mapped mapFile(const std::string& path)
    {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "cannot open " << path << "\n";
            std::exit(2);
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 8)
        {
            std::cerr << path << ": cannot stat or shorter than the 8-byte header\n";
            std::exit(2);
        }
        void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (p == MAP_FAILED)
        {
            std::cerr << "cannot map " << path << "\n";
            std::exit(2);
        }
        return {static_cast<const char*>(p), static_cast<size_t>(st.st_size)};
    }

    struct Matrix
    {
        const float* rows{nullptr};
        int32_t n{0};
        int32_t d{0};
    };

    Matrix mapMatrix(const std::string& path)
    {
        Mapped m = mapFile(path);
        int32_t header[2];
        std::memcpy(header, m.data, sizeof(header));
        if (header[0] < 0 || header[1] <= 0 ||
            m.bytes != 8 + static_cast<size_t>(header[0]) * static_cast<size_t>(header[1]) * sizeof(float))
        {
            std::cerr << path << ": size does not match its header\n";
            std::exit(2);
        }
        return {reinterpret_cast<const float*>(m.data + 8), header[0], header[1]};
    }

    float l2DistanceBytes(uint8_t* a, uint8_t* b, size_t dim)
    {
        return akerL2Distance(reinterpret_cast<float*>(a), reinterpret_cast<float*>(b), dim);
    }

    bool floatCopyTransform(void* src, size_t src_size, size_t dim, void* dst, uint8_t*)
    {
        const size_t need = dim * sizeof(float);
        if (src == nullptr || dst == nullptr || src_size < need)
            return false;
        std::memcpy(dst, src, need);
        return true;
    }

    // Query ids and result ids live in different tables; keep them disjoint and non-zero.
    uint64_t queryId(int64_t universe_id) { return (1ull << 48) | static_cast<uint64_t>(universe_id + 1); }
    uint64_t resultId(int64_t row) { return static_cast<uint64_t>(row) + 1; }

    std::map<std::string, std::string> parseArgs(int argc, char** argv)
    {
        std::map<std::string, std::string> args;
        for (int i = 1; i + 1 < argc; i += 2)
        {
            std::string key = argv[i];
            if (key.rfind("--", 0) != 0)
            {
                std::cerr << "bad argument " << key << "\n";
                std::exit(2);
            }
            args[key.substr(2)] = argv[i + 1];
        }
        return args;
    }

    std::string need(const std::map<std::string, std::string>& a, const std::string& key)
    {
        auto it = a.find(key);
        if (it == a.end())
        {
            std::cerr << "missing --" << key << "\n";
            std::exit(2);
        }
        return it->second;
    }

    std::string opt(const std::map<std::string, std::string>& a, const std::string& key,
                    const std::string& fallback)
    {
        auto it = a.find(key);
        return it == a.end() ? fallback : it->second;
    }

    template <typename T>
    void writeVector(const std::string& path, const std::vector<T>& v)
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
    }
}

int main(int argc, char** argv)
{
    auto args = parseArgs(argc, argv);
    const std::string out_dir = need(args, "out-dir");
    const size_t k = std::stoul(opt(args, "k", "10"));
    const size_t pool_size = std::stoul(need(args, "pool-size"));
    const size_t status_every = std::stoul(opt(args, "status-every", "1000"));

    Matrix base = mapMatrix(need(args, "base"));
    Matrix requests = mapMatrix(need(args, "requests"));
    Mapped stream_file = mapFile(need(args, "stream"));
    Mapped answers_file = mapFile(need(args, "answers"));
    Mapped dists_file = mapFile(need(args, "dists"));
    const int64_t* stream = reinterpret_cast<const int64_t*>(stream_file.data);
    const int64_t* answers = reinterpret_cast<const int64_t*>(answers_file.data);
    const float* dists = reinterpret_cast<const float*>(dists_file.data);

    const size_t m = requests.n;
    const size_t dim = requests.d;
    if (base.d != requests.d || stream_file.bytes != m * 8 || answers_file.bytes != m * k * 8 ||
        dists_file.bytes != m * k * 4)
    {
        std::cerr << "input sizes disagree (base d=" << base.d << ", requests " << m << " x " << dim << ")\n";
        return 2;
    }
    const size_t vector_bytes = dim * sizeof(float);

    anns_cache_parameter_c_t parameter;
    std::memset(&parameter, 0, sizeof(parameter));
    parameter.vector_format.dimension = static_cast<uint32_t>(dim);
    parameter.vector_format.vector_in_bytes = vector_bytes;
    parameter.capacity.pool_size = pool_size;
    parameter.capacity.in_topk = k;
    parameter.capacity.top_delta = std::stoul(opt(args, "top-delta", "0"));
    parameter.tuning.global_thresh = std::stof(opt(args, "global-thresh", "0"));
    parameter.tuning.dropout = 0.0f;
    parameter.tuning.risk_thresh = std::stof(opt(args, "risk-thresh", "0.35"));
    parameter.tuning.alpha_tighten = std::stof(opt(args, "alpha-tighten", "0.25"));
    parameter.tuning.alpha_loosen = std::stof(opt(args, "alpha-loosen", "1.1"));
    parameter.distance_metric = 0;
    parameter.filter.hnsw_m = std::stoi(opt(args, "filter-m", "4"));
    parameter.filter.ef_construction = std::stoi(opt(args, "filter-ef-construction", "8"));
    parameter.filter.ef_search = std::stoi(opt(args, "filter-ef-search", "16"));
#if AKER_ENABLE_PROXIMITY_MODE
    if (parameter.tuning.global_thresh <= 0.0f)
    {
        std::cerr << "proximity mode needs --global-thresh > 0 (squared L2)\n";
        return 2;
    }
#endif

    anns_cache_c_wrapper_t* cache = akerCreateAnnsCache(parameter);
    if (cache == nullptr)
        return 3;

    std::vector<uint8_t> outcome(m, 0), seen_flag(m, 0);
    std::vector<int64_t> result_ids(m * k, -1);
    std::unordered_set<int64_t> seen;
    std::ofstream status_log(out_dir + "/status.log");
    size_t hits_exact = 0, hits_approx = 0, hits_linked = 0, insert_failed = 0, link_failed = 0, invalid = 0;

    for (size_t i = 0; i < m; i++)
    {
        const int64_t uid = stream[i];
        const float* qvec = requests.rows + i * dim;
        seen_flag[i] = seen.count(uid) ? 1 : 0;

        char* q_slot = akerCreateVectorSlot(queryId(uid), vector_bytes,
                                            reinterpret_cast<char*>(const_cast<float*>(qvec)), 0, 0, 0.0f);
        char* q_view = akerCreateVectorView(q_slot, dim, vector_bytes, floatCopyTransform);

        bool similar = false;
        bool is_invalid = false;
        char* hit = akerGetCacheEntry(cache, q_view, &similar, &is_invalid, l2DistanceBytes);
        if (is_invalid)
            invalid++;

        if (hit != nullptr && !is_invalid)
        {
            // Equality only when the entry was stored under this exact vector id; a linked
            // repeat (exact-id lookup that returns another query's entry) is approximate.
            const bool own_entry = akerGetQueryVectorIdFromCacheEntry(hit) == queryId(uid);
            outcome[i] = similar ? 2 : (own_entry ? 1 : 3);
            (similar ? hits_approx : (own_entry ? hits_exact : hits_linked))++;
            for (size_t j = 0; j < k; j++)
            {
                char* slot = akerGetResultVectorSlotAt(hit, static_cast<int>(j));
                result_ids[i * k + j] = static_cast<int64_t>(akerGetVectorIdFromVectorSlot(slot)) - 1;
            }
            if (similar)
            {
                // Mirror the paper's integration: link the new query to the representative.
                const uint64_t found = akerGetQueryVectorIdFromCacheEntry(hit);
                char* dummy = akerCreateCacheEntry(cache, q_slot, k + parameter.capacity.top_delta, nullptr);
                if (!akerLinkCacheEntry(cache, dummy, found))
                {
                    akerDestroyCacheEntry(dummy);
                    link_failed++;
                }
            }
            akerDestroyCacheEntry(hit);
        }
        else
        {
            if (hit != nullptr)
                akerDestroyCacheEntry(hit);
            std::vector<char*> slots;
            slots.reserve(k);
            for (size_t j = 0; j < k; j++)
            {
                const int64_t row = answers[i * k + j];
                if (row < 0 || row >= base.n)
                {
                    std::cerr << "request " << i << ": answer id " << row << " outside the base\n";
                    return 4;
                }
                result_ids[i * k + j] = row;
                const float d = dists[i * k + j];
                slots.push_back(akerCreateVectorSlot(
                    resultId(row), vector_bytes,
                    reinterpret_cast<char*>(const_cast<float*>(base.rows + static_cast<size_t>(row) * dim)),
                    0, 0, d * d));
            }
            char* entry = akerCreateCacheEntry(cache, q_slot, k, slots.data());
            if (!akerInsertCacheEntry(cache, queryId(uid), entry, q_view))
            {
                akerDestroyCacheEntry(entry);
                insert_failed++;
            }
            for (char* slot : slots)
                akerDestroyVectorSlot(slot);
        }
        seen.insert(uid);
        akerDestroyVectorView(q_view);
        akerDestroyVectorSlot(q_slot);

        if (status_every > 0 && (i + 1) % status_every == 0)
            status_log << (i + 1) << " " << akerGetCacheStatusText(cache) << "\n";
    }

    writeVector(out_dir + "/outcome.u8", outcome);
    writeVector(out_dir + "/seen.u8", seen_flag);
    writeVector(out_dir + "/result_ids.i64", result_ids);
    std::ofstream summary(out_dir + "/summary.txt");
    summary << "mode=" <<
#if AKER_ENABLE_PROXIMITY_MODE
        "proximity"
#else
        "standard"
#endif
            << "\nrequests=" << m << "\nhits_exact=" << hits_exact << "\nhits_approx=" << hits_approx << "\nhits_linked=" << hits_linked
            << "\ninsert_failed=" << insert_failed << "\nlink_failed=" << link_failed
            << "\ninvalid=" << invalid << "\nfinal_status=" << akerGetCacheStatusText(cache) << "\n";
    akerDestroyAnnsCache(cache);
    return 0;
}
