// ReCal storage and serving core.
//
// Data pool (full-precision vectors, ids, list counts), shell pool (4-bit codes, no ids), reverse map,
// entries (center, K, j, response list, radius, shell list), an HNSW over the entry centers built once per
// construction, a hash-keyed LRU mini cache, the serving order and byte accounting per structure.
// Construction logic, training and the hit policy live in Python (recal/).
//
// Build: sh recal/core/build.sh

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hnswlib/hnswlib.h>

namespace py = pybind11;

namespace {

constexpr double EPS = 1e-12;
constexpr int LEVELS = 15;  // 4 bits per coordinate

enum Path {
    MISS_K_MAX = 0,
    MINI_HIT = 1,
    MAIN_HIT = 2,
    MISS_SCORE = 3,
    MISS_NO_CANDIDATE = 4,
    MISS_ABSTAIN = 5,
};

uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// 64-bit hash of the vector's bytes (word-wise mix; collisions are ignored)
uint64_t hash_bytes(const float* v, size_t d) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v);
    size_t n = d * sizeof(float);
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ n;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = mix64(h ^ w) + 0x9e3779b97f4a7c15ULL;
    }
    if (i < n) {
        uint64_t w = 0;
        std::memcpy(&w, p + i, n - i);
        h = mix64(h ^ w);
    }
    return mix64(h);
}

double l2(const float* a, const float* b, size_t d) {
    double s = 0;
    for (size_t i = 0; i < d; i++) {
        double t = (double)a[i] - (double)b[i];
        s += t * t;
    }
    return std::sqrt(s);
}

struct Entry {
    std::vector<float> center;
    int32_t K = 0;
    int32_t j = 0;
    float radius = 0;
    std::vector<int32_t> resp;   // data pool slots, ranked by distance to the center
    std::vector<int32_t> shell;  // shell pool slots
    int32_t built_len = 0;       // response length at construction (jK)
};

// hnswlib space over entry numbers: an element is its entry index (4 bytes), a query is a tagged buffer.
// The centers live once, in the entries; the index holds only links, the index and the label.
class CenterSpace : public hnswlib::SpaceInterface<float> {
public:
    CenterSpace(const std::vector<Entry>* entries, size_t dim) : entries_(entries), dim_(dim) {}
    size_t get_data_size() override { return sizeof(int32_t); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return &CenterSpace::dist; }
    void* get_dist_func_param() override { return this; }

    // query buffer: a negative tag followed by the vector
    static std::vector<float> tagged_query(const float* v, size_t dim) {
        std::vector<float> buf(dim + 1);
        int32_t tag = -1;
        std::memcpy(buf.data(), &tag, sizeof(tag));
        std::memcpy(buf.data() + 1, v, dim * sizeof(float));
        return buf;
    }

private:
    const float* vec(const void* p) const {
        int32_t id;
        std::memcpy(&id, p, sizeof(id));
        return id >= 0 ? (*entries_)[id].center.data() : static_cast<const float*>(p) + 1;
    }
    static float dist(const void* a, const void* b, const void* param) {
        const CenterSpace* self = static_cast<const CenterSpace*>(param);
        const float* x = self->vec(a);
        const float* y = self->vec(b);
        float s = 0;
        for (size_t i = 0; i < self->dim_; i++) {
            float t = x[i] - y[i];
            s += t * t;
        }
        return s;
    }
    const std::vector<Entry>* entries_;
    size_t dim_;
};

struct MiniEntry {
    std::vector<int32_t> ids;
    std::vector<float> dists;
};

struct Result {
    int path = MISS_NO_CANDIDATE;
    std::vector<int64_t> ids;
    std::vector<float> dists;
    bool has_score = false;
    double score = std::numeric_limits<double>::quiet_NaN();
    double nr = 0, offset = 0, size = 0, rate = 0;
    int count = 0;
    int best_entry = -1;
    int n_read = 0;
    int n_candidates = 0;
    int n_contrib = -1;  // entries read that list a served vector; -1 unless tracked
};

class Core {
  public:
    Core(int dim, int k_max) : d(dim), k_max_(k_max) {
        if (dim < 1 || k_max < 1) throw std::invalid_argument("dim and k_max must be positive");
        code_bytes_ = (d + 1) / 2;
        lo_.assign(d, 0.f);
        step_.assign(d, 1.f / LEVELS);
    }

    // ---- configuration -------------------------------------------------------------------------

    void set_quantizer(const std::vector<float>& lo, const std::vector<float>& hi) {
        if ((int)lo.size() != d || (int)hi.size() != d) throw std::invalid_argument("quantizer range has wrong length");
        for (int i = 0; i < d; i++) {
            lo_[i] = lo[i];
            step_[i] = (float)(std::max((double)hi[i] - (double)lo[i], EPS) / LEVELS);
        }
        quantizer_set_ = true;
    }

    void set_budget(double bytes) { budget_ = bytes; }
    void set_external_bytes(const std::string& name, double bytes) { external_[name] = bytes; }

    void set_model(const std::vector<double>& mean, const std::vector<double>& scale,
                   const std::vector<double>& coef, double intercept) {
        if (mean.size() != 3 || scale.size() != 3 || coef.size() != 3)
            throw std::invalid_argument("the recall estimator has three features");
        for (int i = 0; i < 3; i++) {
            mean_[i] = mean[i];
            scale_[i] = scale[i] > 0 ? scale[i] : 1.0;
            coef_[i] = coef[i];
        }
        intercept_ = intercept;
        has_model_ = true;
    }

    // ---- construction --------------------------------------------------------------------------

    int add_entry(py::array_t<float, py::array::c_style | py::array::forcecast> center, int K, int j,
                  py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
                  py::array_t<float, py::array::c_style | py::array::forcecast> vecs, double radius) {
        if (center.ndim() != 1 || center.shape(0) != d) throw std::invalid_argument("center has wrong shape");
        if (ids.ndim() != 1 || vecs.ndim() != 2 || vecs.shape(0) != ids.shape(0) || vecs.shape(1) != d)
            throw std::invalid_argument("response ids and vectors do not match");
        if (ids.shape(0) < 1) throw std::invalid_argument("empty response list");
        Entry e;
        e.center.assign(center.data(), center.data() + d);
        e.K = K;
        e.j = j;
        size_t n = ids.shape(0);
        e.resp.reserve(n);
        for (size_t t = 0; t < n; t++) {
            int32_t slot = data_slot(ids.data()[t], vecs.data() + t * d);
            data_count_[slot]++;
            e.resp.push_back(slot);
        }
        e.built_len = (int32_t)n;
        e.radius = radius >= 0 ? (float)radius : (float)l2(e.center.data(), vecs.data() + (n - 1) * d, d);
        entries_.push_back(std::move(e));
        index_ready_ = false;
        inv_ready_ = false;
        return (int)entries_.size() - 1;
    }

    // the shell keeps the listed vectors that are not in the data pool, quantized, shared by id
    int add_shell(int entry, py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
                  py::array_t<float, py::array::c_style | py::array::forcecast> vecs) {
        if (entry < 0 || entry >= (int)entries_.size()) throw std::out_of_range("entry");
        if (ids.ndim() != 1 || vecs.ndim() != 2 || vecs.shape(0) != ids.shape(0) || vecs.shape(1) != d)
            throw std::invalid_argument("shell ids and vectors do not match");
        int added = 0;
        for (py::ssize_t t = 0; t < ids.shape(0); t++) {
            int64_t id = ids.data()[t];
            if (rev_.count(id)) continue;
            int32_t slot;
            auto it = shell_build_map_.find(id);
            if (it == shell_build_map_.end()) {
                slot = (int32_t)(shell_codes_.size() / code_bytes_);
                shell_codes_.resize(shell_codes_.size() + code_bytes_, 0);
                quantize_into(vecs.data() + (size_t)t * d, &shell_codes_[(size_t)slot * code_bytes_]);
                shell_build_map_[id] = slot;
            } else {
                slot = it->second;
            }
            entries_[entry].shell.push_back(slot);
            added++;
        }
        return added;
    }

    void finish_build(int M, int ef_construction, int ef_search, int seed) {
        shell_build_map_.clear();
        shell_build_map_.rehash(0);
        ef_search_ = ef_search;
        index_.reset();
        space_.reset();
        if (!entries_.empty()) {
            space_ = std::make_unique<CenterSpace>(&entries_, d);
            index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), entries_.size(), M,
                                                                       ef_construction, seed);
            for (size_t i = 0; i < entries_.size(); i++) {
                int32_t id = (int32_t)i;
                index_->addPoint(&id, i);
            }
        }
        index_ready_ = true;
        built_ = true;
        built_slots_ = (int)rev_.size();
    }

    // ---- serving -------------------------------------------------------------------------------

    Result lookup(py::array_t<float, py::array::c_style | py::array::forcecast> q, int K,
                  std::optional<double> lam, int m, bool use_mini) {
        check_query(q);
        Result r;
        if (K > k_max_ || K < 1) {
            r.path = MISS_K_MAX;
            return r;
        }
        if (use_mini && mini_capacity_ > 0) {
            if (mini_get(hash_bytes(q.data(), d), K, r)) {
                r.path = MINI_HIT;
                return r;
            }
        }
        main_path(q.data(), K, m, r);
        if (r.n_candidates < K) return r;
        if (!r.has_score) {
            if (lam.has_value()) throw std::runtime_error("the recall estimator is not set");
            r.path = MISS_ABSTAIN;
            return r;
        }
        if (!lam.has_value()) {
            r.path = MISS_ABSTAIN;
            return r;
        }
        r.path = r.score >= *lam ? MAIN_HIT : MISS_SCORE;
        return r;
    }

    // features of the candidate response for many queries: rows (has, nr, offset, size, rate, count, n_read)
    py::tuple candidates_batch(py::array_t<float, py::array::c_style | py::array::forcecast> Q, int K, int m) {
        if (Q.ndim() != 2 || Q.shape(1) != d) throw std::invalid_argument("queries have wrong shape");
        size_t n = Q.shape(0);
        py::array_t<double> feat({(py::ssize_t)n, (py::ssize_t)7});
        py::array_t<int64_t> ids({(py::ssize_t)n, (py::ssize_t)K});
        py::array_t<float> dists({(py::ssize_t)n, (py::ssize_t)K});
        auto f = feat.mutable_unchecked<2>();
        auto I = ids.mutable_unchecked<2>();
        auto D = dists.mutable_unchecked<2>();
        for (size_t i = 0; i < n; i++) {
            Result r;
            main_path(Q.data() + i * d, K, m, r);
            f(i, 0) = r.n_candidates >= K ? 1.0 : 0.0;
            f(i, 1) = r.nr;
            f(i, 2) = r.offset;
            f(i, 3) = r.size;
            f(i, 4) = r.rate;
            f(i, 5) = r.count;
            f(i, 6) = r.n_read;
            for (int t = 0; t < K; t++) {
                bool ok = t < (int)r.ids.size();
                I(i, t) = ok ? r.ids[t] : -1;
                D(i, t) = ok ? r.dists[t] : std::numeric_limits<float>::quiet_NaN();
            }
        }
        return py::make_tuple(feat, ids, dists);
    }

    // entries a query must read (nearest center first) to find every true neighbor the pool holds;
    // -1 when none of its neighbors is in the pool
    std::vector<int> entries_needed_batch(py::array_t<float, py::array::c_style | py::array::forcecast> Q,
                                          py::array_t<int64_t, py::array::c_style | py::array::forcecast> truth) {
        if (Q.ndim() != 2 || Q.shape(1) != d || truth.ndim() != 2 || truth.shape(0) != Q.shape(0))
            throw std::invalid_argument("queries and truth do not match");
        build_inverse();
        size_t n = Q.shape(0), K = truth.shape(1), ne = entries_.size();
        std::vector<int> out(n, -1);
        std::vector<double> dist(ne);
        std::vector<int32_t> order(ne), rank(ne);
        for (size_t i = 0; i < n; i++) {
            std::vector<int32_t> slots;
            for (size_t t = 0; t < K; t++) {
                auto it = rev_.find(truth.data()[i * K + t]);
                if (it != rev_.end() && !inv_[it->second].empty()) slots.push_back(it->second);
            }
            if (slots.empty()) continue;
            for (size_t e = 0; e < ne; e++) dist[e] = l2(Q.data() + i * d, entries_[e].center.data(), d);
            for (size_t e = 0; e < ne; e++) order[e] = (int32_t)e;
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return dist[a] < dist[b]; });
            for (size_t r = 0; r < ne; r++) rank[order[r]] = (int32_t)r;
            int need = 0;
            for (int32_t s : slots) {
                int best = (int)ne;
                for (int32_t e : inv_[s]) best = std::min(best, (int)rank[e]);
                need = std::max(need, best + 1);
            }
            out[i] = need;
        }
        return out;
    }

    // ---- mini cache ----------------------------------------------------------------------------

    uint64_t hash_vector(py::array_t<float, py::array::c_style | py::array::forcecast> q) {
        check_query(q);
        return hash_bytes(q.data(), d);
    }

    void mini_set_capacity(size_t R) {
        mini_capacity_ = R;
        while (mini_lru_.size() > mini_capacity_) mini_evict();
    }

    void mini_put(py::array_t<float, py::array::c_style | py::array::forcecast> q,
                  py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
                  py::array_t<float, py::array::c_style | py::array::forcecast> dists) {
        check_query(q);
        if (ids.ndim() != 1 || dists.ndim() != 1 || ids.shape(0) != dists.shape(0))
            throw std::invalid_argument("response ids and distances do not match");
        if (mini_capacity_ == 0) return;
        uint64_t h = hash_bytes(q.data(), d);
        MiniEntry e;
        e.ids = ids_to_int32(ids.data(), ids.shape(0));
        e.dists.assign(dists.data(), dists.data() + dists.shape(0));
        auto it = mini_map_.find(h);
        if (it != mini_map_.end()) {
            it->second->second = std::move(e);
            mini_lru_.splice(mini_lru_.begin(), mini_lru_, it->second);
            return;
        }
        if (mini_lru_.size() >= mini_capacity_) mini_evict();
        mini_lru_.emplace_front(h, std::move(e));
        mini_map_[h] = mini_lru_.begin();
    }

    // an entry fetched for K' >= K answers with its first K results
    py::object mini_get_py(py::array_t<float, py::array::c_style | py::array::forcecast> q, int K) {
        check_query(q);
        Result r;
        if (!mini_get(hash_bytes(q.data(), d), K, r)) return py::none();
        return py::make_tuple(py::array_t<int64_t>(r.ids.size(), r.ids.data()),
                              py::array_t<float>(r.dists.size(), r.dists.data()));
    }

    // the mini cache survives a construction: dump (oldest first) and load by hash
    py::list mini_dump() const {
        py::list out;
        for (auto it = mini_lru_.rbegin(); it != mini_lru_.rend(); ++it)
            out.append(py::make_tuple(it->first, widen(it->second.ids),
                                      py::array_t<float>(it->second.dists.size(), it->second.dists.data())));
        return out;
    }

    void mini_put_hash(uint64_t h, py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
                       py::array_t<float, py::array::c_style | py::array::forcecast> dists) {
        if (mini_capacity_ == 0) return;
        MiniEntry e;
        e.ids = ids_to_int32(ids.data(), ids.shape(0));
        e.dists.assign(dists.data(), dists.data() + dists.shape(0));
        auto it = mini_map_.find(h);
        if (it != mini_map_.end()) {
            it->second->second = std::move(e);
            mini_lru_.splice(mini_lru_.begin(), mini_lru_, it->second);
            return;
        }
        if (mini_lru_.size() >= mini_capacity_) mini_evict();
        mini_lru_.emplace_front(h, std::move(e));
        mini_map_[h] = mini_lru_.begin();
    }

    // hit by hash (no query vector needed): used when a stored response is looked up by key
    py::object mini_get_hash(uint64_t h, int K) {
        Result r;
        if (!mini_get(h, K, r)) return py::none();
        return py::make_tuple(py::array_t<int64_t>(r.ids.size(), r.ids.data()),
                              py::array_t<float>(r.dists.size(), r.dists.data()));
    }

    // allocated bytes of the stored ids and distances (the 8K part of the entry size)
    size_t mini_payload_bytes() const {
        size_t b = 0;
        for (const auto& kv : mini_lru_)
            b += kv.second.ids.capacity() * sizeof(int32_t) + kv.second.dists.capacity() * sizeof(float);
        return b;
    }

    // bytes per element of the center index outside its links: independent of the dimension
    size_t center_index_payload_per_element() const {
        return index_ ? index_->size_data_per_element_ - index_->size_links_level0_ : 0;
    }

    size_t mini_size() const { return mini_lru_.size(); }
    size_t mini_capacity() const { return mini_capacity_; }

    // ---- updates --------------------------------------------------------------------

    // remove stale pointers and free the dead slots (called once 5% of the pool has been deleted)
    int sweep() {
        size_t n = data_id_.size();
        std::vector<int32_t> remap(n, -1);
        int32_t next = 0;
        for (size_t s = 0; s < n; s++)
            if (data_alive_[s]) remap[s] = next++;
        int removed = (int)n - next;
        if (removed == 0) return 0;
        std::vector<float> vec;
        std::vector<int64_t> id;
        vec.reserve((size_t)next * d);
        for (size_t s = 0; s < n; s++) {
            if (!data_alive_[s]) continue;
            vec.insert(vec.end(), data_vec_.begin() + s * d, data_vec_.begin() + (s + 1) * d);
            id.push_back(data_id_[s]);
        }
        data_vec_.swap(vec);
        data_id_.swap(id);
        data_count_.assign(next, 0);
        data_alive_.assign(next, 1);
        for (auto& kv : rev_) kv.second = remap[kv.second];
        for (Entry& e : entries_) {
            std::vector<int32_t> keep;
            for (int32_t s : e.resp)
                if (remap[s] >= 0) keep.push_back(remap[s]);
            e.resp.swap(keep);
            for (int32_t s : e.resp) data_count_[s]++;
        }
        dstamp_.assign(next, 0);
        depoch_ = 0;
        inv_ready_ = false;
        return removed;
    }

    int n_built_slots() const { return built_slots_; }

    // distance from an entry's center to its farthest live member
    double farthest_live(const Entry& e) const {
        for (size_t t = e.resp.size(); t-- > 0;)
            if (data_alive_[e.resp[t]]) return l2(e.center.data(), &data_vec_[(size_t)e.resp[t] * d], d);
        return 0.0;
    }

    int live_len(const Entry& e) const {
        int c = 0;
        for (int32_t s : e.resp) c += data_alive_[s] ? 1 : 0;
        return c;
    }

    // lazy batched inserts. Returns (admitted, frozen, ignored, headroom_left)
    py::tuple insert_batch(py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
                           py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
                           double headroom_left, int rounds, int per_round) {
        if (ids.ndim() != 1 || vecs.ndim() != 2 || vecs.shape(0) != ids.shape(0) || vecs.shape(1) != d)
            throw std::invalid_argument("insert ids and vectors do not match");
        int admitted = 0, frozen = 0, ignored = 0;
        for (py::ssize_t t = 0; t < ids.shape(0); t++) {
            const float* v = vecs.data() + (size_t)t * d;
            int64_t id = ids.data()[t];
            if (rev_.count(id)) {
                ignored++;
                continue;
            }
            std::vector<int> joins = containing_entries(v, rounds, per_round);
            if (joins.empty()) {
                ignored++;
                continue;
            }
            // lists that deletes shortened below their built length take the insert without displacement;
            // the new vector still counts against the headroom
            std::vector<char> direct(joins.size());
            for (size_t i = 0; i < joins.size(); i++) direct[i] = live_len(entries_[joins[i]]) < entries_[joins[i]].built_len;
            double added = data_slot_bytes(d) + reverse_map_bytes() + 4.0 * joins.size();
            if (added <= headroom_left) {
                int32_t slot = new_slot(id, v);
                for (int e : joins) join_list(e, slot, v);
                headroom_left -= added;
                admitted++;
                continue;
            }
            // headroom used: admit only if displacing the farthest member of each list frees as many slots
            std::unordered_map<int32_t, int> drops;
            std::vector<int32_t> farthest(joins.size(), -1);
            for (size_t i = 0; i < joins.size(); i++) {
                if (direct[i]) continue;
                const Entry& e = entries_[joins[i]];
                for (size_t k = e.resp.size(); k-- > 0;)
                    if (data_alive_[e.resp[k]]) {
                        farthest[i] = e.resp[k];
                        break;
                    }
                if (farthest[i] >= 0) drops[farthest[i]]++;
            }
            int freed = 0;
            for (const auto& kv : drops)
                if (data_count_[kv.first] - kv.second <= 0) freed++;
            if (freed < 1) {
                frozen++;
                continue;
            }
            int32_t reuse = -1;
            for (size_t i = 0; i < joins.size(); i++) {
                if (farthest[i] < 0) continue;
                Entry& e = entries_[joins[i]];
                auto pos = std::find(e.resp.begin(), e.resp.end(), farthest[i]);
                if (pos != e.resp.end()) e.resp.erase(pos);
                data_count_[farthest[i]]--;
            }
            for (const auto& kv : drops)
                if (data_count_[kv.first] <= 0 && reuse < 0) reuse = kv.first;
            int32_t slot;
            if (reuse >= 0) {
                rev_.erase(data_id_[reuse]);
                std::copy(v, v + d, data_vec_.begin() + (size_t)reuse * d);
                data_id_[reuse] = id;
                data_alive_[reuse] = 1;
                data_count_[reuse] = 0;
                rev_[id] = reuse;
                slot = reuse;
            } else {
                slot = new_slot(id, v);
            }
            for (int e : joins) join_list(e, slot, v);
            for (const auto& kv : drops)
                if (data_count_[kv.first] <= 0 && kv.first != slot) {
                    data_alive_[kv.first] = 0;
                    rev_.erase(data_id_[kv.first]);
                }
            for (int e : joins) {
                Entry& en = entries_[e];
                // the radius shrinks to the new last member of a list that lost its farthest member
                for (size_t i = 0; i < joins.size(); i++)
                    if (joins[i] == e && farthest[i] >= 0) en.radius = (float)farthest_live(en);
            }
            admitted++;
        }
        inv_ready_ = false;
        return py::make_tuple(admitted, frozen, ignored, headroom_left);
    }

    // ---- accounting and inspection -------------------------------------------------------------

    py::dict bytes_by_structure() const {
        double data = 0, shell = 0, entries = 0, rev = 0, mini = 0, index = 0;
        size_t live = rev_.size();
        data = (double)data_id_.size() * (coord_bytes_ * d + 8 + 4);  // dead slots stay until a sweep
        shell = (double)(shell_codes_.size() / code_bytes_) * code_bytes_ + (quantizer_set_ ? 8.0 * d : 0.0);
        for (const Entry& e : entries_)
            entries += coord_bytes_ * d + 20 + 4.0 * (e.resp.size() + e.shell.size());
        rev = (double)live * 12.0;
        for (const auto& kv : mini_lru_) mini += mini_entry_bytes((int)kv.second.ids.size());
        if (index_) {
            // allocated size of the index: level-0 blocks (links, 4-byte index, 8-byte label), link-list pointers, levels and upper links;
            // the label map and the visited-list pool are not counted (a few percent of the index)
            size_t n = index_->cur_element_count;
            double links = (double)index_->max_elements_ * (index_->size_data_per_element_ + sizeof(void*) + sizeof(int));
            for (size_t i = 0; i < n; i++) links += (double)index_->element_levels_[i] * index_->size_links_per_element_;
            index = links;
        }
        py::dict out;
        out["data_pool"] = data;
        out["shell_pool"] = shell;
        out["entries"] = entries;
        out["reverse_map"] = rev;
        out["center_index"] = index;
        out["recall_estimator"] = has_model_ ? 80.0 : 0.0;
        out["mini_cache"] = mini;
        out["headroom"] = 0.02 * budget_;
        double total = data + shell + entries + rev + index + (has_model_ ? 80.0 : 0.0) + mini + 0.02 * budget_;
        for (const auto& kv : external_) {
            out[py::str(kv.first)] = kv.second;
            total += kv.second;
        }
        out["total"] = total;
        out["budget"] = budget_;
        return out;
    }

    py::dict entry(int i) const {
        if (i < 0 || i >= (int)entries_.size()) throw std::out_of_range("entry");
        const Entry& e = entries_[i];
        std::vector<int64_t> ids;
        for (int32_t s : e.resp) ids.push_back(data_id_[s]);
        py::dict out;
        out["K"] = e.K;
        out["j"] = e.j;
        out["radius"] = (double)e.radius;
        out["response_ids"] = ids;
        out["shell_slots"] = e.shell;
        out["built_len"] = e.built_len;
        return out;
    }

    int n_entries() const { return (int)entries_.size(); }
    void set_track_contributing(bool on) { track_contrib_ = on; }
    void set_coord_bytes(double b) { coord_bytes_ = b; }
    std::vector<int32_t> entry_sizes() const {
        std::vector<int32_t> out;
        out.reserve(entries_.size());
        for (const auto& e : entries_) out.push_back((int32_t)e.resp.size());
        return out;
    }
    int n_data_slots() const { return (int)rev_.size(); }  // live slots
    int n_dead_slots() const { return (int)(data_id_.size() - rev_.size()); }

    // eager delete; the slot is marked dead and leaves the reverse map, list pointers stay
    bool delete_id(int64_t id) {
        auto it = rev_.find(id);
        if (it == rev_.end()) return false;
        data_alive_[it->second] = 0;
        rev_.erase(it);
        inv_ready_ = false;
        return true;
    }
    int n_shell_slots() const { return (int)(shell_codes_.size() / code_bytes_); }
    int dim() const { return d; }
    int k_max() const { return k_max_; }

    static py::array_t<int64_t> widen(const std::vector<int32_t>& v) {
        py::array_t<int64_t> out(v.size());
        for (size_t i = 0; i < v.size(); i++) out.mutable_at(i) = v[i];
        return out;
    }

    static std::vector<int32_t> ids_to_int32(const int64_t* p, py::ssize_t n) {
        std::vector<int32_t> out(n);
        for (py::ssize_t i = 0; i < n; i++) {
            if (p[i] < INT32_MIN || p[i] > INT32_MAX) throw std::invalid_argument("id does not fit in 4 bytes");
            out[i] = (int32_t)p[i];
        }
        return out;
    }

    static double mini_entry_bytes(int K) { return 40.0 + 8.0 * K; }
    static double data_slot_bytes(int d) { return 4.0 * d + 12.0; }
    static double shell_slot_bytes(int d) { return (double)((d + 1) / 2); }
    static double entry_bytes(int d, int resp_len, int shell_len) {
        return 4.0 * d + 20.0 + 4.0 * (resp_len + shell_len);
    }
    static double reverse_map_bytes() { return 12.0; }

  private:
    int d;
    int k_max_;
    int code_bytes_;
    bool built_ = false, index_ready_ = false, inv_ready_ = false, quantizer_set_ = false, has_model_ = false;
    int ef_search_ = 64;
    int built_slots_ = 0;
    double budget_ = 0;
    std::unordered_map<std::string, double> external_;

    std::vector<float> lo_, step_;
    std::vector<float> data_vec_;
    std::vector<int64_t> data_id_;
    std::vector<int32_t> data_count_;
    std::vector<uint8_t> data_alive_;
    std::unordered_map<int64_t, int32_t> rev_;
    std::vector<uint8_t> shell_codes_;
    std::unordered_map<int64_t, int32_t> shell_build_map_;  // transient
    std::vector<Entry> entries_;
    std::unique_ptr<CenterSpace> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    std::vector<std::vector<int32_t>> inv_;  // data slot -> entries listing it

    double mean_[3] = {0, 0, 0}, scale_[3] = {1, 1, 1}, coef_[3] = {0, 0, 0}, intercept_ = 0;

    std::vector<uint32_t> dstamp_, sstamp_, kstamp_;
    uint32_t depoch_ = 0, sepoch_ = 0, kepoch_ = 0;
    bool track_contrib_ = false;
    double coord_bytes_ = 4.0;  // bytes charged per stored vector coordinate

    size_t mini_capacity_ = 0;
    std::list<std::pair<uint64_t, MiniEntry>> mini_lru_;
    std::unordered_map<uint64_t, std::list<std::pair<uint64_t, MiniEntry>>::iterator> mini_map_;

    int32_t new_slot(int64_t id, const float* v) {
        int32_t slot = (int32_t)data_id_.size();
        data_vec_.insert(data_vec_.end(), v, v + d);
        data_id_.push_back(id);
        data_count_.push_back(0);
        data_alive_.push_back(1);
        rev_[id] = slot;
        return slot;
    }

    // keep the list ranked by distance to the center
    void join_list(int e, int32_t slot, const float* v) {
        Entry& en = entries_[e];
        double dn = l2(en.center.data(), v, d);
        size_t pos = en.resp.size();
        while (pos > 0) {
            int32_t s = en.resp[pos - 1];
            if (data_alive_[s] && l2(en.center.data(), &data_vec_[(size_t)s * d], d) <= dn) break;
            pos--;
        }
        en.resp.insert(en.resp.begin() + pos, slot);
        data_count_[slot]++;
    }

    // entries whose sphere holds v, found through the center index: per_round centers per round,
    // up to `rounds` rounds, stopping after a round that adds no entry
    std::vector<int> containing_entries(const float* v, int rounds, int per_round) {
        std::vector<int> found;
        size_t ne = entries_.size();
        if (ne == 0) return found;
        std::vector<char> seen(ne, 0);
        for (int r = 1; r <= rounds; r++) {
            size_t k = std::min<size_t>((size_t)r * per_round, ne);
            std::vector<std::pair<double, int>> near;
            if (index_) {
                index_->setEf(std::max<size_t>(ef_search_, k));
                auto tq = CenterSpace::tagged_query(v, d);
                auto pq = index_->searchKnn(tq.data(), k);
                while (!pq.empty()) {
                    near.emplace_back(0.0, (int)pq.top().second);
                    pq.pop();
                }
            } else {
                for (size_t e = 0; e < ne; e++) near.emplace_back(l2(v, entries_[e].center.data(), d), (int)e);
                std::sort(near.begin(), near.end());
                near.resize(k);
            }
            bool changed = false;
            for (const auto& pr : near) {
                int e = pr.second;
                if (seen[e]) continue;
                seen[e] = 1;
                if (l2(v, entries_[e].center.data(), d) <= (double)entries_[e].radius) {
                    found.push_back(e);
                    changed = true;
                }
            }
            if (!changed || k >= ne) break;
        }
        return found;
    }

    void check_query(const py::array_t<float, py::array::c_style | py::array::forcecast>& q) const {
        if (q.ndim() != 1 || q.shape(0) != d) throw std::invalid_argument("query has wrong shape");
    }

    int32_t data_slot(int64_t id, const float* vec) {
        auto it = rev_.find(id);
        if (it != rev_.end()) return it->second;
        int32_t slot = (int32_t)data_id_.size();
        data_vec_.insert(data_vec_.end(), vec, vec + d);
        data_id_.push_back(id);
        data_count_.push_back(0);
        data_alive_.push_back(1);
        rev_[id] = slot;
        return slot;
    }

    void quantize_into(const float* x, uint8_t* out) const {
        for (int i = 0; i < d; i++) {
            double c = std::round(((double)x[i] - lo_[i]) / step_[i]);
            int code = (int)std::min<double>(LEVELS, std::max<double>(0, c));
            if (i % 2 == 0) out[i / 2] = (uint8_t)code;
            else out[i / 2] |= (uint8_t)(code << 4);
        }
    }

    // squared distance from q to a shell code, stopping once it reaches `limit`
    double shell_sq_dist(const float* q, const uint8_t* code, double limit) const {
        double s = 0;
        for (int i = 0; i < d; i++) {
            int c = (i % 2 == 0) ? (code[i / 2] & 15) : (code[i / 2] >> 4);
            double t = (double)q[i] - ((double)lo_[i] + c * (double)step_[i]);
            s += t * t;
            if (s >= limit) return s;
        }
        return s;
    }

    void mini_evict() {
        mini_map_.erase(mini_lru_.back().first);
        mini_lru_.pop_back();
    }

    bool mini_get(uint64_t h, int K, Result& r) {
        auto it = mini_map_.find(h);
        if (it == mini_map_.end()) return false;
        const MiniEntry& e = it->second->second;
        if ((int)e.ids.size() < K) return false;
        mini_lru_.splice(mini_lru_.begin(), mini_lru_, it->second);
        r.ids.assign(e.ids.begin(), e.ids.begin() + K);
        r.dists.assign(e.dists.begin(), e.dists.begin() + K);
        return true;
    }

    void build_inverse() {
        if (inv_ready_) return;
        inv_.assign(data_id_.size(), {});
        for (size_t e = 0; e < entries_.size(); e++)
            for (int32_t s : entries_[e].resp)
                if (data_alive_[s]) inv_[s].push_back((int32_t)e);
        inv_ready_ = true;
    }

    // the m entries with the nearest centers, nearest first, with exact center distances
    void nearest_entries(const float* q, int m, std::vector<std::pair<double, int>>& out) {
        out.clear();
        size_t ne = entries_.size();
        if (ne == 0) return;
        size_t want = (m <= 0 || (size_t)m >= ne) ? ne : (size_t)m;
        if (want == ne || !index_) {
            for (size_t e = 0; e < ne; e++) out.emplace_back(l2(q, entries_[e].center.data(), d), (int)e);
            std::stable_sort(out.begin(), out.end());
            out.resize(want);
            return;
        }
        index_->setEf(std::max<size_t>(ef_search_, want));
        auto tq = CenterSpace::tagged_query(q, d);
        auto pq = index_->searchKnn(tq.data(), want);
        while (!pq.empty()) {
            int e = (int)pq.top().second;
            out.emplace_back(l2(q, entries_[e].center.data(), d), e);
            pq.pop();
        }
        std::stable_sort(out.begin(), out.end());
    }

    void count_contributing(const std::vector<std::pair<double, int32_t>>& cand, int K,
                            const std::vector<std::pair<double, int>>& read, Result& r) {
        if (kstamp_.size() < data_id_.size()) kstamp_.resize(data_id_.size(), 0);
        if (++kepoch_ == 0) {
            std::fill(kstamp_.begin(), kstamp_.end(), 0);
            kepoch_ = 1;
        }
        for (int t = 0; t < K; t++) kstamp_[cand[t].second] = kepoch_;
        int n = 0;
        for (const auto& re : read) {
            for (int32_t s : entries_[re.second].resp) {
                if (kstamp_[s] == kepoch_) {
                    n++;
                    break;
                }
            }
        }
        r.n_contrib = n;
    }

    void main_path(const float* q, int K, int m, Result& r) {
        std::vector<std::pair<double, int>> read;
        nearest_entries(q, m, read);
        r.n_read = (int)read.size();
        r.path = MISS_NO_CANDIDATE;
        if (read.empty()) return;
        if (dstamp_.size() < data_id_.size()) dstamp_.resize(data_id_.size(), 0);
        if (++depoch_ == 0) {
            std::fill(dstamp_.begin(), dstamp_.end(), 0);
            depoch_ = 1;
        }
        std::vector<std::pair<double, int32_t>> cand;
        for (const auto& re : read) {
            for (int32_t s : entries_[re.second].resp) {
                if (!data_alive_[s] || dstamp_[s] == depoch_) continue;
                dstamp_[s] = depoch_;
                cand.emplace_back(l2(q, &data_vec_[(size_t)s * d], d), s);
            }
        }
        r.n_candidates = (int)cand.size();
        if ((int)cand.size() < K) return;
        std::partial_sort(cand.begin(), cand.begin() + K, cand.end());
        r.ids.resize(K);
        r.dists.resize(K);
        for (int t = 0; t < K; t++) {
            r.ids[t] = data_id_[cand[t].second];
            r.dists[t] = (float)cand[t].first;
        }
        if (track_contrib_) count_contributing(cand, K, read, r);
        double nr = std::max(cand[K - 1].first, EPS);
        // best-fit entry among those read
        double best = std::numeric_limits<double>::infinity();
        int best_e = -1;
        double best_de = 0, best_rad = 1;
        for (const auto& re : read) {
            double rad = std::max((double)entries_[re.second].radius, EPS);
            double fit = (re.first + nr) / rad;
            if (fit < best) {
                best = fit;
                best_e = re.second;
                best_de = re.first;
                best_rad = rad;
            }
        }
        r.best_entry = best_e;
        r.nr = nr;
        r.offset = best_de / best_rad;
        r.size = nr / best_rad;
        // intruders: shell vectors of every entry read, once per shell slot, strictly closer than nr
        size_t ns = shell_codes_.size() / code_bytes_;
        if (sstamp_.size() < ns) sstamp_.resize(ns, 0);
        if (++sepoch_ == 0) {
            std::fill(sstamp_.begin(), sstamp_.end(), 0);
            sepoch_ = 1;
        }
        double nr2 = nr * nr;
        int count = 0;
        for (const auto& re : read) {
            for (int32_t s : entries_[re.second].shell) {
                if (sstamp_[s] == sepoch_) continue;
                sstamp_[s] = sepoch_;
                if (shell_sq_dist(q, &shell_codes_[(size_t)s * code_bytes_], nr2) < nr2) count++;
            }
        }
        r.count = count;
        r.rate = std::log((count + 1.0) / K);
        if (has_model_) {
            double z = intercept_;
            double f[3] = {r.offset, r.size, r.rate};
            for (int i = 0; i < 3; i++) z += coef_[i] * (f[i] - mean_[i]) / scale_[i];
            r.score = 1.0 / (1.0 + std::exp(-z));
            r.has_score = true;
        } else {
            r.has_score = false;
        }
        r.path = MISS_SCORE;  // overwritten by the caller when a decision is made
    }
};

}  // namespace

PYBIND11_MODULE(_recal_core, mod) {
    mod.doc() = "ReCal storage and serving core";
    py::class_<Result>(mod, "Result")
        .def_readonly("path", &Result::path)
        .def_property_readonly("ids", [](const Result& r) { return py::array_t<int64_t>(r.ids.size(), r.ids.data()); })
        .def_property_readonly("dists", [](const Result& r) { return py::array_t<float>(r.dists.size(), r.dists.data()); })
        .def_readonly("has_score", &Result::has_score)
        .def_readonly("score", &Result::score)
        .def_readonly("nr", &Result::nr)
        .def_readonly("offset", &Result::offset)
        .def_readonly("size", &Result::size)
        .def_readonly("rate", &Result::rate)
        .def_readonly("count", &Result::count)
        .def_readonly("best_entry", &Result::best_entry)
        .def_readonly("n_read", &Result::n_read)
        .def_readonly("n_candidates", &Result::n_candidates)
        .def_readonly("n_contrib", &Result::n_contrib);
    mod.attr("MISS_K_MAX") = (int)MISS_K_MAX;
    mod.attr("MINI_HIT") = (int)MINI_HIT;
    mod.attr("MAIN_HIT") = (int)MAIN_HIT;
    mod.attr("MISS_SCORE") = (int)MISS_SCORE;
    mod.attr("MISS_NO_CANDIDATE") = (int)MISS_NO_CANDIDATE;
    mod.attr("MISS_ABSTAIN") = (int)MISS_ABSTAIN;
    py::class_<Core>(mod, "Core")
        .def(py::init<int, int>(), py::arg("dim"), py::arg("k_max"))
        .def("set_quantizer", &Core::set_quantizer)
        .def("set_budget", &Core::set_budget)
        .def("set_track_contributing", &Core::set_track_contributing)
        .def("set_coord_bytes", &Core::set_coord_bytes)
        .def("entry_sizes", [](const Core& c) { auto v = c.entry_sizes(); return py::array_t<int32_t>(v.size(), v.data()); })
        .def("set_external_bytes", &Core::set_external_bytes)
        .def("set_model", &Core::set_model)
        .def("add_entry", &Core::add_entry, py::arg("center"), py::arg("K"), py::arg("j"), py::arg("ids"),
             py::arg("vecs"), py::arg("radius") = -1.0)
        .def("add_shell", &Core::add_shell)
        .def("finish_build", &Core::finish_build, py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("ef_search") = 64, py::arg("seed") = 100)
        .def("lookup", &Core::lookup, py::arg("q"), py::arg("K"), py::arg("lam") = py::none(), py::arg("m") = 0,
             py::arg("use_mini") = true)
        .def("candidates_batch", &Core::candidates_batch, py::arg("Q"), py::arg("K"), py::arg("m") = 0)
        .def("entries_needed_batch", &Core::entries_needed_batch)
        .def("hash_vector", &Core::hash_vector)
        .def("mini_set_capacity", &Core::mini_set_capacity)
        .def("mini_put", &Core::mini_put)
        .def("mini_get", &Core::mini_get_py)
        .def("mini_dump", &Core::mini_dump)
        .def("mini_put_hash", &Core::mini_put_hash)
        .def("mini_get_hash", &Core::mini_get_hash)
        .def("mini_payload_bytes", &Core::mini_payload_bytes)
        .def("center_index_payload_per_element", &Core::center_index_payload_per_element)
        .def("mini_size", &Core::mini_size)
        .def("mini_capacity", &Core::mini_capacity)
        .def("bytes_by_structure", &Core::bytes_by_structure)
        .def("entry", &Core::entry)
        .def("n_entries", &Core::n_entries)
        .def("n_data_slots", &Core::n_data_slots)
        .def("n_dead_slots", &Core::n_dead_slots)
        .def("delete_id", &Core::delete_id)
        .def("sweep", &Core::sweep)
        .def("n_built_slots", &Core::n_built_slots)
        .def("insert_batch", &Core::insert_batch, py::arg("ids"), py::arg("vecs"), py::arg("headroom_left"),
             py::arg("rounds") = 5, py::arg("per_round") = 16)
        .def("n_shell_slots", &Core::n_shell_slots)
        .def_property_readonly("dim", &Core::dim)
        .def_property_readonly("k_max", &Core::k_max)
        .def_static("mini_entry_bytes", &Core::mini_entry_bytes)
        .def_static("data_slot_bytes", &Core::data_slot_bytes)
        .def_static("shell_slot_bytes", &Core::shell_slot_bytes)
        .def_static("entry_bytes", &Core::entry_bytes)
        .def_static("reverse_map_bytes", &Core::reverse_map_bytes);
}
