// Cache core (doc 9ah): the request path of our cache, shared by the headline (simulator) and timing runs.
//
// State, maintained incrementally (no rebuild per change):
//   cached vectors   refcounted by database id (an id is cached while some entry's ball holds it); an
//                    HNSW graph over them (hnswlib; evicted ids are marked deleted, revived ids unmarked;
//                    rebuild_index() when the deleted share grows) and a dense store for exact search
//   shell samples    refcounted by database id (the union of the entries' sampled shells), stored as given
//                    (already quantized by the caller); a sample counts as an intruder candidate only while
//                    its id is not cached (the simulator's shell = union of shells minus cached vectors)
//   entries          center vector, radius, ball ids, shell sample ids, last use (LRU)
// Request path, lookup(x):
//   knn    the k nearest cached vectors: exact scan (ef1 = 0) or HNSW at search list ef1 (stage 1);
//          neighborhood radius nr = distance to the k-th (+1e-12)
//   balls  fit = (nr + d(x, center)) / radius for every entry; features of the 3 best-fitting balls:
//          d / radius and nr / radius (POOR_FIT when fewer entries)
//   shell  intruders: samples (not cached) within nr, over the whole sample (shell_balls = 0) or over the
//          shells of the shell_balls best-fitting entries (a sample shared by several counts once)
//   model  score = ((f - mu) / sd) . coef + b with f[6] = log1p(intruders); hit iff score >= lambda
//   serve  on a hit, with ef2 > 0: the k nearest cached vectors at search list ef2 (stage 2)
// Exact squared distances in float32 with float accumulation (as FAISS / hnswlib).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "third_party/hnswlib/hnswlib.h"

namespace py = pybind11;
using i64 = int64_t;

static const int N_FEAT = 7;
static const int N_BALLS = 3;
static const double POOR_FIT = 10.0;
static const double EPS = 1e-12;

static inline float l2sq(const float* a, const float* b, int d) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < d; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct LookupResult {
    bool hit = false;
    double score = -std::numeric_limits<double>::infinity();
    double nr = 0.0;
    int intruders = 0;
    std::vector<i64> near;          // stage-1 neighbors; on a hit with stage 2: the served points
    std::vector<double> features;   // 7
    double t_knn = 0, t_balls = 0, t_shell = 0, t_model = 0, t_serve = 0;   // seconds
};

class Core {
public:
    Core(int dim, int k, int shell_balls, int M, int ef_construction, size_t capacity, int seed, bool graph)
        : d_(dim), k_(k), shell_balls_(shell_balls), M_(M), efc_(ef_construction), seed_(seed), graph_on_(graph),
          space_(dim) {
        cap_ = std::max<size_t>(capacity, 1024);
        if (graph_on_) hnsw_.reset(new hnswlib::HierarchicalNSW<float>(&space_, cap_, M_, efc_, seed_));
        mu_.assign(N_FEAT, 0.0);
        sd_.assign(N_FEAT, 1.0);
        coef_.assign(N_FEAT, 0.0);
    }

    // ---------------------------------------------------------------- model / settings
    void set_model(std::vector<double> mu, std::vector<double> sd, std::vector<double> coef, double b) {
        if (mu.size() != N_FEAT || sd.size() != N_FEAT || coef.size() != N_FEAT)
            throw std::invalid_argument("model needs 7 features");
        mu_ = mu; sd_ = sd; coef_ = coef; b_ = b;
    }
    void set_shell_balls(int n) { shell_balls_ = n; }

    // Shell samples as packed 4-bit codes (doc 9ai): sample values are lo + c * step per dimension (the
    // simulator's b-bit quantization); stored as two codes per byte instead of the decoded float32 values,
    // distances computed from the codes (identical counts, 1/8 of the memory read). Set before any sample.
    void set_quantizer(std::vector<double> lo, std::vector<double> hi, int bits) {
        if (!sid_.empty() && sslot_.size()) throw std::logic_error("set_quantizer: samples present");
        if (bits != 4) { qpack_ = false; return; }
        if ((int)lo.size() != d_ || (int)hi.size() != d_) throw std::invalid_argument("quantizer dims");
        qpack_ = true;
        nb_ = (d_ + 1) / 2;
        qlo_.assign(d_, 0.f); qstep_.assign(d_, 0.f);
        for (int i = 0; i < d_; ++i) { qlo_[i] = (float)lo[i]; qstep_[i] = (float)(std::max(hi[i] - lo[i], 1e-12) / 15.0); }
        se_.assign(nb_, 0.f); so_.assign(nb_, 0.f);
        for (int j = 0; j < nb_; ++j) { se_[j] = qstep_[2 * j]; if (2 * j + 1 < d_) so_[j] = qstep_[2 * j + 1]; }
        sstore_.clear(); scode_.clear(); sid_.clear(); sref_.clear(); scached_.clear(); sstamp_.clear(); sfree_.clear();
        sslot_.clear(); n_samp_eff_ = 0;
    }

    // How the balls of a request are found (doc 9ai): 0 = every entry, best fit (default); 1 = best fit among
    // the m nearest centers (graph over the centers at search list ef); 2 = the nearest centers themselves
    // (3 for the score features, shell_balls for the intruder count), no fit ordering
    void set_ball_mode(int mode, int m, int ef) {
        ball_mode_ = mode; ball_m_ = std::max(m, N_BALLS); ball_ef_ = std::max(ef, ball_m_);
        if (mode > 0 && !cidx_) rebuild_center_index();
    }
    void set_threads(int n) { threads_ = std::max(1, n); }

    // ---------------------------------------------------------------- cached vectors
    // ids may already be cached (refcount + 1); vecs: one row per id
    void add_cached(py::array_t<i64, py::array::c_style | py::array::forcecast> ids,
                    py::array_t<float, py::array::c_style | py::array::forcecast> vecs) {
        auto I = ids.unchecked<1>();
        auto V = vecs.unchecked<2>();
        if (V.shape(0) != I.shape(0) || V.shape(1) != d_) throw std::invalid_argument("vecs shape");
        std::vector<std::pair<i64, int>> fresh;      // (id, slot) needing a graph insert
        for (ssize_t j = 0; j < I.shape(0); ++j) {
            i64 id = I(j);
            auto it = vslot_.find(id);
            int s;
            if (it == vslot_.end()) {
                s = new_vslot(id, &V(j, 0));
                fresh.push_back({id, s});
            } else {
                s = it->second;
            }
            if (vref_[s]++ == 0) {
                ++n_cached_;
                on_cached_change(id, true);
                if (it != vslot_.end() && graph_on_) {    // revived: was evicted, still in the graph
                    if (in_graph_[s]) { hnsw_->unmarkDelete((size_t)id); --n_deleted_; }
                    else fresh.push_back({id, s});
                }
            }
        }
        graph_insert(fresh);
    }

    void remove_cached(py::array_t<i64, py::array::c_style | py::array::forcecast> ids) {
        auto I = ids.unchecked<1>();
        for (ssize_t j = 0; j < I.shape(0); ++j) drop_vector(I(j));
    }

    // ---------------------------------------------------------------- shell samples
    void add_samples(py::array_t<i64, py::array::c_style | py::array::forcecast> ids,
                     py::array_t<float, py::array::c_style | py::array::forcecast> vecs) {
        auto I = ids.unchecked<1>();
        auto V = vecs.unchecked<2>();
        if (V.shape(0) != I.shape(0) || V.shape(1) != d_) throw std::invalid_argument("sample vecs shape");
        for (ssize_t j = 0; j < I.shape(0); ++j) add_sample(I(j), &V(j, 0));
    }

    void remove_samples(py::array_t<i64, py::array::c_style | py::array::forcecast> ids) {
        auto I = ids.unchecked<1>();
        for (ssize_t j = 0; j < I.shape(0); ++j) drop_sample(I(j));
    }

    // ---------------------------------------------------------------- entries
    // center: the entry's query vector; ball: its cached ids (added to the cached set here); shell: sample
    // ids (with their stored vectors in shell_vecs; added to the sample pool here)
    void add_entry(i64 c, py::array_t<float, py::array::c_style | py::array::forcecast> center, double radius,
                   py::array_t<i64, py::array::c_style | py::array::forcecast> ball,
                   py::array_t<float, py::array::c_style | py::array::forcecast> ball_vecs,
                   py::array_t<i64, py::array::c_style | py::array::forcecast> shell,
                   py::array_t<float, py::array::c_style | py::array::forcecast> shell_vecs, double t) {
        if (crow_.count(c)) throw std::invalid_argument("entry exists");
        auto C = center.unchecked<1>();
        if (C.shape(0) != d_) throw std::invalid_argument("center dim");
        add_cached(ball, ball_vecs);
        add_samples(shell, shell_vecs);
        int r = (int)ec_.size();
        crow_[c] = r;
        ec_.push_back(c);
        cstore_.insert(cstore_.end(), &C(0), &C(0) + d_);
        radius_.push_back(std::max(radius, EPS));
        auto B = ball.unchecked<1>();
        std::vector<i64> bv(B.shape(0));
        for (ssize_t j = 0; j < B.shape(0); ++j) { bv[j] = B(j); holders_[B(j)].push_back(c); }
        eball_.push_back(std::move(bv));
        auto S = shell.unchecked<1>();
        std::vector<int> sv(S.shape(0));
        for (ssize_t j = 0; j < S.shape(0); ++j) sv[j] = sslot_.at(S(j));
        eshell_.push_back(std::move(sv));
        std::vector<i64> sid(S.shape(0));
        for (ssize_t j = 0; j < S.shape(0); ++j) sid[j] = S(j);
        eshell_ids_.push_back(std::move(sid));
        used_.push_back(t);
        if (cidx_) center_insert(c, &C(0));
    }

    void remove_entry(i64 c) {
        auto it = crow_.find(c);
        if (it == crow_.end()) throw std::invalid_argument("no such entry");
        int r = it->second, last = (int)ec_.size() - 1;
        for (i64 v : eball_[r]) {
            auto& h = holders_[v];
            h.erase(std::find(h.begin(), h.end(), c));
            if (h.empty()) holders_.erase(v);
            drop_vector(v);
        }
        for (i64 s : eshell_ids_[r]) drop_sample(s);
        if (r != last) {
            std::copy(cstore_.begin() + (size_t)last * d_, cstore_.begin() + (size_t)(last + 1) * d_,
                      cstore_.begin() + (size_t)r * d_);
            radius_[r] = radius_[last];
            ec_[r] = ec_[last];
            eball_[r] = std::move(eball_[last]);
            eshell_[r] = std::move(eshell_[last]);
            eshell_ids_[r] = std::move(eshell_ids_[last]);
            used_[r] = used_[last];
            crow_[ec_[r]] = r;
        }
        cstore_.resize((size_t)last * d_);
        radius_.pop_back(); ec_.pop_back(); eball_.pop_back(); eshell_.pop_back(); eshell_ids_.pop_back();
        used_.pop_back();
        crow_.erase(c);
        if (cidx_) {
            cidx_->markDelete((size_t)c);
            ++c_deleted_;
            if (c_deleted_ > 1000 && c_deleted_ > ec_.size() / 2) rebuild_center_index();
        }
    }

    void center_insert(i64 c, const float* v) {
        if (cidx_labels_.count(c)) { cidx_->unmarkDelete((size_t)c); --c_deleted_; return; }
        if (cidx_->getCurrentElementCount() + 1 > ccap_) { ccap_ *= 2; cidx_->resizeIndex(ccap_); }
        cidx_->addPoint(v, (size_t)c);
        cidx_labels_.insert(c);
    }

    void rebuild_center_index() {
        ccap_ = std::max<size_t>(2 * ec_.size() + 1024, 4096);
        cidx_.reset(new hnswlib::HierarchicalNSW<float>(&space_, ccap_, 16, 100, seed_));
        cidx_labels_.clear();
        c_deleted_ = 0;
        for (size_t r = 0; r < ec_.size(); ++r) center_insert(ec_[r], &cstore_[r * d_]);
    }

    bool has_entry(i64 c) const { return crow_.count(c) > 0; }

    // ---------------------------------------------------------------- LRU bookkeeping
    void mark_used(py::array_t<i64, py::array::c_style | py::array::forcecast> ids, double t) {
        auto I = ids.unchecked<1>();
        for (ssize_t j = 0; j < I.shape(0); ++j) {
            auto it = holders_.find(I(j));
            if (it == holders_.end()) continue;
            for (i64 c : it->second) used_[crow_[c]] = t;
        }
    }
    void set_used(i64 c, double t) { used_[crow_.at(c)] = t; }
    double last_used(i64 c) const { return used_[crow_.at(c)]; }
    // entry centers, least recently used first
    std::vector<i64> lru_order() const {
        std::vector<int> idx(ec_.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return used_[a] < used_[b]; });
        std::vector<i64> out(idx.size());
        for (size_t i = 0; i < idx.size(); ++i) out[i] = ec_[idx[i]];
        return out;
    }

    // ---------------------------------------------------------------- sizes
    size_t n_cached() const { return n_cached_; }
    size_t n_samples() const { return n_samp_eff_; }      // samples whose id is not cached
    size_t n_entries() const { return ec_.size(); }
    double memory(int bits) const { return (double)n_cached_ + (double)n_samp_eff_ * bits / 32.0; }
    size_t graph_elements() const { return hnsw_ ? hnsw_->getCurrentElementCount() : 0; }
    size_t dead_slots() const { return vid_.size() - vfree_.size() - n_cached_; }
    size_t graph_deleted() const { return n_deleted_; }
    std::vector<i64> entries() const { return ec_; }
    std::vector<i64> cached_ids() const {
        std::vector<i64> out;
        out.reserve(n_cached_);
        for (size_t s = 0; s < vid_.size(); ++s) if (vref_[s] > 0) out.push_back(vid_[s]);
        return out;
    }
    double radius(i64 c) const { return radius_[crow_.at(c)]; }

    // rebuild the graph over the live cached vectors (background work in a real system)
    void rebuild_index(int threads) {
        cap_ = std::max<size_t>(cap_, 2 * n_cached_ + 1024);
        if (graph_on_) hnsw_.reset(new hnswlib::HierarchicalNSW<float>(&space_, cap_, M_, efc_, seed_));
        std::fill(in_graph_.begin(), in_graph_.end(), 0);
        n_deleted_ = 0;
        for (size_t s = 0; s < vid_.size(); ++s)          // evicted vectors leave the store
            if (vref_[s] == 0 && vid_[s] >= 0) { vslot_.erase(vid_[s]); vid_[s] = -1; vfree_.push_back((int)s); }
        std::vector<std::pair<i64, int>> all;
        for (size_t s = 0; s < vid_.size(); ++s) if (vref_[s] > 0) all.push_back({vid_[s], (int)s});
        int keep = threads_;
        threads_ = std::max(1, threads);
        graph_insert(all);
        threads_ = keep;
    }

    // ---------------------------------------------------------------- request path
    LookupResult lookup(py::array_t<float, py::array::c_style | py::array::forcecast> x, double lam, int ef1, int ef2) {
        auto X = x.unchecked<1>();
        if (X.shape(0) != d_) throw std::invalid_argument("query dim");
        return lookup_ptr(&X(0), lam, ef1, ef2, threads_);
    }

    // a batch with no cache change in between (static caches, headline batches): parallel over queries
    py::tuple lookup_batch(py::array_t<float, py::array::c_style | py::array::forcecast> xs, double lam, int ef1, int ef2) {
        auto Xs = xs.unchecked<2>();
        if (Xs.shape(1) != d_) throw std::invalid_argument("query dim");
        ssize_t n = Xs.shape(0);
        std::vector<LookupResult> res(n);
        const float* base = &Xs(0, 0);
        {
            py::gil_scoped_release nogil;
            // phase 1: decisions (stage 1); hnswlib's search list is global, so it is set once per phase
            if (ef1 > 0) hnsw_->setEf(std::max(ef1, k_));
            if (cidx_ && ball_mode_ > 0) cidx_->setEf(std::max<size_t>(ball_ef_, std::max<size_t>(ball_m_, std::max(N_BALLS, shell_balls_))));
            parallel_for(n, threads_, [&](ssize_t i) { res[i] = lookup_ptr(base + i * d_, lam, ef1, 0, 1, true); });
            // phase 2: served points of the hits (stage 2)
            if (ef2 > 0) {
                hnsw_->setEf(std::max(ef2, k_));
                parallel_for(n, threads_, [&](ssize_t i) {
                    if (!res[i].hit) return;
                    std::vector<std::pair<float, i64>> nn2;
                    graph_knn(base + i * d_, ef2, nn2, true);
                    if ((int)nn2.size() < k_) res[i].hit = false;
                    else for (int j = 0; j < k_; ++j) res[i].near[j] = nn2[j].second;
                });
            }
        }
        py::array_t<bool> hit(n);
        py::array_t<double> score(n), nr(n);
        py::array_t<i64> near({(ssize_t)n, (ssize_t)k_});
        py::array_t<double> feat({(ssize_t)n, (ssize_t)N_FEAT});
        auto H = hit.mutable_unchecked<1>(); auto S = score.mutable_unchecked<1>(); auto R = nr.mutable_unchecked<1>();
        auto N = near.mutable_unchecked<2>(); auto F = feat.mutable_unchecked<2>();
        for (ssize_t i = 0; i < n; ++i) {
            H(i) = res[i].hit; S(i) = res[i].score; R(i) = res[i].nr;
            for (int j = 0; j < k_; ++j) N(i, j) = j < (int)res[i].near.size() ? res[i].near[j] : -1;
            for (int j = 0; j < N_FEAT; ++j) F(i, j) = res[i].features[j];
        }
        return py::make_tuple(hit, score, near, nr, feat);
    }

    // stage 2 alone (labels of misses need the would-be served answer)
    std::vector<i64> serve(py::array_t<float, py::array::c_style | py::array::forcecast> x, int ef2) {
        auto X = x.unchecked<1>();
        std::vector<std::pair<float, i64>> nn;
        if (ef2 > 0) graph_knn(&X(0), ef2, nn); else exact_knn(&X(0), nn, threads_);
        std::vector<i64> out;
        for (auto& p : nn) out.push_back(p.second);
        return out;
    }

private:
    // ---------------------------------------------------------------- vector / sample slots
    int new_vslot(i64 id, const float* v) {
        int s;
        if (!vfree_.empty()) { s = vfree_.back(); vfree_.pop_back(); vid_[s] = id; vref_[s] = 0; in_graph_[s] = 0;
                               std::copy(v, v + d_, vstore_.begin() + (size_t)s * d_); }
        else { s = (int)vid_.size(); vid_.push_back(id); vref_.push_back(0); in_graph_.push_back(0);
               vstore_.insert(vstore_.end(), v, v + d_); }
        vslot_[id] = s;
        return s;
    }

    void drop_vector(i64 id) {
        auto it = vslot_.find(id);
        if (it == vslot_.end()) throw std::invalid_argument("vector not cached");
        int s = it->second;
        if (vref_[s] <= 0) throw std::logic_error("vector refcount");
        if (--vref_[s] == 0) {
            --n_cached_;
            on_cached_change(id, false);
            if (graph_on_ && in_graph_[s]) { hnsw_->markDelete((size_t)id); ++n_deleted_; }
            // the slot keeps the vector (and the graph node) for a cheap revival; a rebuild drops it
        }
    }

    void encode(const float* v, uint8_t* out) const {
        for (int j = 0; j < nb_; ++j) {
            int c0 = (int)std::lrint((v[2 * j] - qlo_[2 * j]) / qstep_[2 * j]);
            int c1 = 2 * j + 1 < d_ ? (int)std::lrint((v[2 * j + 1] - qlo_[2 * j + 1]) / qstep_[2 * j + 1]) : 0;
            c0 = std::min(std::max(c0, 0), 15); c1 = std::min(std::max(c1, 0), 15);
            out[j] = (uint8_t)(c0 | (c1 << 4));
        }
    }

    void add_sample(i64 id, const float* v) {
        auto it = sslot_.find(id);
        int s;
        if (it == sslot_.end()) {
            if (!sfree_.empty()) { s = sfree_.back(); sfree_.pop_back(); sid_[s] = id; sref_[s] = 0; sstamp_[s] = 0;
                                   if (qpack_) encode(v, &scode_[(size_t)s * nb_]);
                                   else std::copy(v, v + d_, sstore_.begin() + (size_t)s * d_); }
            else { s = (int)sid_.size(); sid_.push_back(id); sref_.push_back(0); scached_.push_back(0);
                   if (qpack_) { scode_.resize(scode_.size() + nb_); encode(v, &scode_[(size_t)s * nb_]); }
                   else sstore_.insert(sstore_.end(), v, v + d_);
                   sstamp_.push_back(0); }
            sslot_[id] = s;
            auto vt = vslot_.find(id);
            scached_[s] = (vt != vslot_.end() && vref_[vt->second] > 0) ? 1 : 0;
        } else {
            s = it->second;
        }
        if (sref_[s]++ == 0 && !scached_[s]) ++n_samp_eff_;
    }

    void drop_sample(i64 id) {
        auto it = sslot_.find(id);
        if (it == sslot_.end()) throw std::invalid_argument("sample not present");
        int s = it->second;
        if (--sref_[s] == 0) {
            if (!scached_[s]) --n_samp_eff_;
            sslot_.erase(it);
            sfree_.push_back(s);
        }
    }

    void on_cached_change(i64 id, bool cached) {
        auto it = sslot_.find(id);
        if (it == sslot_.end()) return;
        int s = it->second;
        scached_[s] = cached ? 1 : 0;
        if (sref_[s] > 0) n_samp_eff_ += cached ? -1 : 1;
    }

    // ---------------------------------------------------------------- graph
    void graph_insert(const std::vector<std::pair<i64, int>>& items) {
        if (items.empty() || !graph_on_) return;      // exact mode (headline): no graph to maintain
        size_t need = hnsw_->getCurrentElementCount() + items.size();
        if (need > cap_) { cap_ = std::max(need, 2 * cap_); hnsw_->resizeIndex(cap_); }
        // bulk builds (warm-up, rebuilds) at ef_construction; incremental inserts (an entry's ball) at 40, as
        // the Python harness did (2x cheaper, measured 11-11): background work either way
        hnsw_->ef_construction_ = std::max<size_t>(items.size() > 256 ? (size_t)efc_ : std::min(40, efc_), (size_t)M_);
        auto add = [&](ssize_t j) {
            hnsw_->addPoint(&vstore_[(size_t)items[j].second * d_], (size_t)items[j].first);
            in_graph_[items[j].second] = 1;
        };
        if (threads_ > 1 && items.size() > 256) {
            py::gil_scoped_release nogil;
            parallel_for((ssize_t)items.size(), threads_, add);
        } else {
            for (ssize_t j = 0; j < (ssize_t)items.size(); ++j) add(j);
        }
    }

    void graph_knn(const float* x, int ef, std::vector<std::pair<float, i64>>& out, bool ef_set = false) const {
        if (!ef_set) hnsw_->setEf(std::max(ef, k_));
        auto pq = hnsw_->searchKnn(x, k_);
        out.resize(pq.size());
        for (int i = (int)pq.size() - 1; i >= 0; --i) { out[i] = {pq.top().first, (i64)pq.top().second}; pq.pop(); }
    }

    void exact_knn(const float* x, std::vector<std::pair<float, i64>>& out, int threads) const {
        size_t n = vid_.size();
        auto scan = [&](size_t lo, size_t hi, std::vector<std::pair<float, i64>>& heap) {
            for (size_t s = lo; s < hi; ++s) {
                if (vref_[s] <= 0) continue;
                float dd = l2sq(x, &vstore_[s * d_], d_);
                if ((int)heap.size() < k_) { heap.push_back({dd, vid_[s]}); std::push_heap(heap.begin(), heap.end()); }
                else if (dd < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end()); heap.back() = {dd, vid_[s]}; std::push_heap(heap.begin(), heap.end());
                }
            }
        };
        std::vector<std::pair<float, i64>> heap;
        if (threads > 1 && n > 20000) {
            std::vector<std::vector<std::pair<float, i64>>> parts(threads);
            std::vector<std::thread> th;
            for (int t = 0; t < threads; ++t)
                th.emplace_back([&, t] { scan(n * t / threads, n * (t + 1) / threads, parts[t]); });
            for (auto& h : th) h.join();
            for (auto& p : parts) for (auto& e : p) {
                if ((int)heap.size() < k_) { heap.push_back(e); std::push_heap(heap.begin(), heap.end()); }
                else if (e.first < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end()); heap.back() = e; std::push_heap(heap.begin(), heap.end());
                }
            }
        } else {
            scan(0, n, heap);
        }
        std::sort(heap.begin(), heap.end());
        out = heap;
    }

    // ---------------------------------------------------------------- one lookup
    LookupResult lookup_ptr(const float* x, double lam, int ef1, int ef2, int threads, bool ef_set = false) {
        LookupResult r;
        r.features.assign(N_FEAT, POOR_FIT);
        double t0 = now_s();
        std::vector<std::pair<float, i64>> nn;
        if (n_cached_ < (size_t)k_) {        // fewer than k cached vectors: nothing can be served (scored as the simulator)
            r.features[N_FEAT - 1] = 0.0;
            r.score = model_score(r.features);
            return r;
        }
        if (ef1 > 0 && !graph_on_) throw std::invalid_argument("graph search on a core built without graph");
        if (ef1 > 0) graph_knn(x, ef1, nn, ef_set); else exact_knn(x, nn, threads);
        double t1 = now_s();
        r.near.resize(nn.size());
        for (size_t i = 0; i < nn.size(); ++i) r.near[i] = nn[i].second;
        if ((int)nn.size() < k_) { r.t_knn = t1 - t0; return r; }
        double nr = std::sqrt(std::max((double)nn.back().first, 0.0)) + EPS;
        r.nr = nr;
        // balls
        size_t ne = ec_.size();
        std::vector<double> fit(ne), de(ne);
        size_t nb = std::min<size_t>(N_BALLS, ne);
        size_t ns = shell_balls_ > 0 ? std::min<size_t>(shell_balls_, ne) : 0;
        size_t top = std::max(nb, ns);
        std::vector<int> best;
        auto by_fit = [&](int a, int b) { return fit[a] < fit[b] || (fit[a] == fit[b] && a < b); };
        if (ball_mode_ == 0 || !cidx_) {
            for (size_t e = 0; e < ne; ++e) {
                de[e] = std::sqrt(std::max((double)l2sq(x, &cstore_[e * d_], d_), 0.0));
                fit[e] = (nr + de[e]) / radius_[e];
            }
            std::vector<int> idx(ne);
            for (size_t e = 0; e < ne; ++e) idx[e] = (int)e;
            if (top < ne) std::nth_element(idx.begin(), idx.begin() + top, idx.end(), by_fit);
            best.assign(idx.begin(), idx.begin() + top);
            std::sort(best.begin(), best.end(), by_fit);
        } else {
            size_t want = ball_mode_ == 1 ? std::max<size_t>(ball_m_, top) : top;
            if (!ef_set) cidx_->setEf(std::max<size_t>(ball_ef_, want));
            auto pq = cidx_->searchKnn(x, std::min(want, ne));
            std::vector<int> cand;
            while (!pq.empty()) {
                int e = crow_.at((i64)pq.top().second);
                de[e] = std::sqrt(std::max((double)pq.top().first, 0.0));
                fit[e] = (nr + de[e]) / radius_[e];
                cand.push_back(e);
                pq.pop();
            }
            if (ball_mode_ == 1) std::sort(cand.begin(), cand.end(), by_fit);
            else std::sort(cand.begin(), cand.end(), [&](int a, int b) { return de[a] < de[b] || (de[a] == de[b] && a < b); });
            if (cand.size() > top) cand.resize(top);
            best = cand;
            nb = std::min(nb, best.size());
            ns = std::min(ns, best.size());
        }
        for (size_t j = 0; j < nb; ++j) {
            int e = best[j];
            r.features[2 * j] = de[e] / radius_[e];
            r.features[2 * j + 1] = nr / radius_[e];
        }
        double t2 = now_s();
        // shell
        float nr2 = (float)(nr * nr);
        int cnt = 0;
        QPrep qp;
        if (qpack_) qprep(x, qp);
        if (shell_balls_ <= 0) {
            for (size_t s = 0; s < sid_.size(); ++s)
                if (sref_[s] > 0 && !scached_[s] && sample_d2(x, qp, (int)s) < nr2) ++cnt;
        } else if (!ef_set) {             // one lookup at a time: one stamp array
            uint32_t stamp = next_stamp();
            for (size_t j = 0; j < ns; ++j)
                for (int s : eshell_[best[j]]) {
                    if (sstamp_[s] == stamp) continue;
                    sstamp_[s] = stamp;
                    if (!scached_[s] && sample_d2(x, qp, s) < nr2) ++cnt;
                }
        } else {                          // parallel batch: local dedupe
            std::vector<int> sl;
            for (size_t j = 0; j < ns; ++j) sl.insert(sl.end(), eshell_[best[j]].begin(), eshell_[best[j]].end());
            std::sort(sl.begin(), sl.end());
            sl.erase(std::unique(sl.begin(), sl.end()), sl.end());
            for (int s : sl) if (!scached_[s] && sample_d2(x, qp, s) < nr2) ++cnt;
        }
        r.intruders = cnt;
        r.features[N_FEAT - 1] = std::log1p((double)cnt);
        double t3 = now_s();
        double sc = model_score(r.features);
        r.score = sc;
        // lambda -inf: every request takes the hit path (the would-be served answer, for labels / tuning)
        r.hit = std::isfinite(sc) && (std::isinf(lam) && lam < 0 ? true : (std::isfinite(lam) && sc >= lam));
        double t4 = now_s();
        if (r.hit && ef2 > 0) {
            std::vector<std::pair<float, i64>> nn2;
            graph_knn(x, ef2, nn2);
            if (ef1 > 0 && ef_set) hnsw_->setEf(std::max(ef1, k_));
            if ((int)nn2.size() < k_) r.hit = false;
            else for (int i = 0; i < k_; ++i) r.near[i] = nn2[i].second;
        }
        double t5 = now_s();
        r.t_knn = t1 - t0; r.t_balls = t2 - t1; r.t_shell = t3 - t2; r.t_model = t4 - t3; r.t_serve = t5 - t4;
        return r;
    }

    // per-request data for the packed-code distance: a = x - lo, split into even / odd dimensions
    struct QPrep { std::vector<float> ae, ao; };
    void qprep(const float* x, QPrep& p) const {
        p.ae.assign(nb_, 0.f); p.ao.assign(nb_, 0.f);
        for (int j = 0; j < nb_; ++j) {
            p.ae[j] = x[2 * j] - qlo_[2 * j];
            if (2 * j + 1 < d_) p.ao[j] = x[2 * j + 1] - qlo_[2 * j + 1];
        }
    }
    inline float sample_d2(const float* x, const QPrep& p, int s) const {
        if (!qpack_) return l2sq(x, &sstore_[(size_t)s * d_], d_);
        const uint8_t* c = &scode_[(size_t)s * nb_];
        const float *ae = p.ae.data(), *ao = p.ao.data(), *se = se_.data(), *so = so_.data();
        float acc = 0.f;
#pragma omp simd reduction(+ : acc)
        for (int j = 0; j < nb_; ++j) {
            float t0 = ae[j] - se[j] * (float)(c[j] & 15);
            float t1 = ao[j] - so[j] * (float)(c[j] >> 4);
            acc += t0 * t0 + t1 * t1;
        }
        return acc;
    }

    double model_score(const std::vector<double>& f) const {
        double sc = b_;
        for (int j = 0; j < N_FEAT; ++j) sc += (f[j] - mu_[j]) / sd_[j] * coef_[j];
        return sc;
    }

    uint32_t next_stamp() {
        // single-threaded shell scans use one stamp array; batch lookups (parallel) use the whole-sample
        // mode or are run with one thread when shell_balls > 0
        if (++stamp_ == 0) { std::fill(sstamp_.begin(), sstamp_.end(), 0); stamp_ = 1; }
        return stamp_;
    }

    template <class F>
    static void parallel_for(ssize_t n, int threads, F f) {
        if (threads <= 1 || n < 2) { for (ssize_t i = 0; i < n; ++i) f(i); return; }
        std::vector<std::thread> th;
        std::atomic<ssize_t> next(0);
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&] { for (ssize_t i; (i = next.fetch_add(1)) < n;) f(i); });
        for (auto& h : th) h.join();
    }

    int d_, k_, shell_balls_, M_, efc_, seed_, threads_ = 1;
    bool graph_on_ = true;
    // packed 4-bit shell samples
    bool qpack_ = false;
    int nb_ = 0;
    std::vector<float> qlo_, qstep_, se_, so_;
    std::vector<uint8_t> scode_;
    // center index (ball modes 1 / 2)
    int ball_mode_ = 0, ball_m_ = 64, ball_ef_ = 64;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> cidx_;
    std::unordered_set<i64> cidx_labels_;
    size_t ccap_ = 0, c_deleted_ = 0;
    size_t cap_;
    hnswlib::L2Space space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
    std::vector<double> mu_, sd_, coef_;
    double b_ = 0.0;
    // cached vectors (slots)
    std::unordered_map<i64, int> vslot_;
    std::vector<i64> vid_;
    std::vector<int> vref_;
    std::vector<char> in_graph_;
    std::vector<float> vstore_;
    std::vector<int> vfree_;
    size_t n_cached_ = 0, n_deleted_ = 0;
    // shell samples (slots)
    std::unordered_map<i64, int> sslot_;
    std::vector<i64> sid_;
    std::vector<int> sref_;
    std::vector<char> scached_;
    std::vector<float> sstore_;
    std::vector<uint32_t> sstamp_;
    std::vector<int> sfree_;
    uint32_t stamp_ = 0;
    size_t n_samp_eff_ = 0;
    // entries (rows)
    std::unordered_map<i64, int> crow_;
    std::vector<i64> ec_;
    std::vector<float> cstore_;
    std::vector<double> radius_;
    std::vector<std::vector<i64>> eball_;
    std::vector<std::vector<int>> eshell_;
    std::vector<std::vector<i64>> eshell_ids_;
    std::vector<double> used_;
    std::unordered_map<i64, std::vector<i64>> holders_;
};

#include "qvcache.h"

PYBIND11_MODULE(_cachecore, m) {
    m.doc() = "Cache core: incremental cache state and the two-stage hit path (doc 9ah)";
    py::class_<LookupResult>(m, "LookupResult")
        .def_readonly("hit", &LookupResult::hit)
        .def_readonly("score", &LookupResult::score)
        .def_readonly("nr", &LookupResult::nr)
        .def_readonly("intruders", &LookupResult::intruders)
        .def_readonly("near", &LookupResult::near)
        .def_readonly("features", &LookupResult::features)
        .def_readonly("t_knn", &LookupResult::t_knn)
        .def_readonly("t_balls", &LookupResult::t_balls)
        .def_readonly("t_shell", &LookupResult::t_shell)
        .def_readonly("t_model", &LookupResult::t_model)
        .def_readonly("t_serve", &LookupResult::t_serve);
    py::class_<Core>(m, "Core")
        .def(py::init<int, int, int, int, int, size_t, int, bool>(), py::arg("dim"), py::arg("k"), py::arg("shell_balls"),
             py::arg("M") = 32, py::arg("ef_construction") = 200, py::arg("capacity") = 1 << 16, py::arg("seed") = 100,
             py::arg("graph") = true)
        .def("set_model", &Core::set_model)
        .def("set_shell_balls", &Core::set_shell_balls)
        .def("set_quantizer", &Core::set_quantizer)
        .def("set_ball_mode", &Core::set_ball_mode, py::arg("mode"), py::arg("m") = 64, py::arg("ef") = 64)
        .def("set_threads", &Core::set_threads)
        .def("add_cached", &Core::add_cached)
        .def("remove_cached", &Core::remove_cached)
        .def("add_samples", &Core::add_samples)
        .def("remove_samples", &Core::remove_samples)
        .def("add_entry", &Core::add_entry, py::arg("c"), py::arg("center"), py::arg("radius"), py::arg("ball"),
             py::arg("ball_vecs"), py::arg("shell"), py::arg("shell_vecs"), py::arg("t") = 0.0)
        .def("remove_entry", &Core::remove_entry)
        .def("has_entry", &Core::has_entry)
        .def("mark_used", &Core::mark_used)
        .def("set_used", &Core::set_used)
        .def("last_used", &Core::last_used)
        .def("lru_order", &Core::lru_order)
        .def("n_cached", &Core::n_cached)
        .def("n_samples", &Core::n_samples)
        .def("n_entries", &Core::n_entries)
        .def("memory", &Core::memory)
        .def("graph_elements", &Core::graph_elements)
        .def("graph_deleted", &Core::graph_deleted)
        .def("dead_slots", &Core::dead_slots)
        .def("entries", &Core::entries)
        .def("cached_ids", &Core::cached_ids)
        .def("radius", &Core::radius)
        .def("rebuild_index", &Core::rebuild_index)
        .def("lookup", &Core::lookup, py::arg("x"), py::arg("lam"), py::arg("ef1"), py::arg("ef2") = 0)
        .def("lookup_batch", &Core::lookup_batch, py::arg("xs"), py::arg("lam"), py::arg("ef1"), py::arg("ef2") = 0)
        .def("serve", &Core::serve);
    py::class_<QVCore>(m, "QVCore")
        .def(py::init<int, int, int, size_t, double, double, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, int, bool, int, int, int>(),
             py::arg("dim"), py::arg("k"), py::arg("n_mini"), py::arg("cap"), py::arg("D"), py::arg("alpha"),
             py::arg("mu"), py::arg("P"), py::arg("lo"), py::arg("hi"), py::arg("nb"), py::arg("graph"),
             py::arg("M") = 16, py::arg("ef_construction") = 64, py::arg("ef") = 64)
        .def("lookup", &QVCore::lookup)
        .def("on_miss", &QVCore::on_miss)
        .def("n_cached", &QVCore::n_cached)
        .def("n_minis_used", &QVCore::n_minis_used)
        .def("n_regions", &QVCore::n_regions)
        .def("cached_ids", &QVCore::cached_ids);
}
