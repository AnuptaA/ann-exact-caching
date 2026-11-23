// QVCache's request path in C++ (doc 9ah), the same algorithm as online_baselines.QVCache (simulator) and
// eval_timing_online.QVCacheFast (timing, diskannpy mini-indexes):
//   n_mini mini-indexes of cached vectors (each up to cap vectors); a vector belongs to one mini-index
//   region(x)  bucket of each of the first `dims` PCA coordinates of x (PCA fit on the warm-up queries,
//              range from them, `nb` buckets per coordinate)
//   lookup(x)  the k nearest cached vectors over all mini-indexes (exact scan, or each mini-index's graph
//              at search list ef, merged); hit iff >= k found, the region has a threshold theta, and the
//              k-th distance <= (1 + D) * theta; a hit marks the mini-indexes that supplied the answer used
//   on_miss    theta(region) <- the backend's k-th distance (first time) or (1 - alpha) theta + alpha d_k;
//              the answer's vectors not cached yet go into the current mini-index; when it would overflow,
//              the next is an empty one, else the least recently used one, cleared
// Graph mode uses hnswlib (M 16 = graph degree 32 at the base layer, efConstruction 64, search list 64:
// the mini-index settings of the diskannpy version, complexity 64 / degree 32), the library of our core.
#pragma once
#include <cstdint>
#include <unordered_map>

class QVCore {
public:
    QVCore(int dim, int k, int n_mini, size_t cap, double D, double alpha,
           py::array_t<double, py::array::c_style | py::array::forcecast> mu,
           py::array_t<double, py::array::c_style | py::array::forcecast> P,
           py::array_t<double, py::array::c_style | py::array::forcecast> lo,
           py::array_t<double, py::array::c_style | py::array::forcecast> hi,
           int nb, bool graph, int M, int efc, int ef)
        : d_(dim), k_(k), n_(n_mini), cap_(cap), D_(D), alpha_(alpha), nb_(nb), graph_(graph), M_(M), efc_(efc),
          ef_(ef), space_(dim), minis_(n_mini), used_(n_mini, 0.0) {
        auto m = mu.unchecked<1>();
        auto p = P.unchecked<2>();
        if (m.shape(0) != d_ || p.shape(0) != d_) throw std::invalid_argument("PCA shapes");
        dims_ = (int)p.shape(1);
        if (dims_ * 3 > 63 || nb_ > 8) throw std::invalid_argument("region key: dims * 3 bits, <= 8 buckets");
        mu_.assign(&m(0), &m(0) + d_);
        P_.resize((size_t)d_ * dims_);
        for (int i = 0; i < d_; ++i) for (int j = 0; j < dims_; ++j) P_[(size_t)i * dims_ + j] = p(i, j);
        auto L = lo.unchecked<1>(); auto H = hi.unchecked<1>();
        lo_.assign(&L(0), &L(0) + dims_);
        hi_.assign(&H(0), &H(0) + dims_);
    }

    uint64_t region(const float* x) const {
        uint64_t key = 0;
        for (int j = 0; j < dims_; ++j) {
            double z = 0.0;
            for (int i = 0; i < d_; ++i) z += ((double)x[i] - mu_[i]) * P_[(size_t)i * dims_ + j];
            double b = std::floor((z - lo_[j]) / std::max(hi_[j] - lo_[j], 1e-12) * nb_);
            int bi = (int)std::min(std::max(b, 0.0), (double)(nb_ - 1));
            key = key * 8 + (uint64_t)bi;
        }
        return key;
    }

    // (hit, near ids, k-th distance of the cached answer)
    py::tuple lookup(py::array_t<float, py::array::c_style | py::array::forcecast> x, double t) {
        auto X = x.unchecked<1>();
        const float* q = &X(0);
        std::vector<std::pair<float, i64>> all;
        for (auto& mi : minis_) {
            size_t n = mi.ids.size();
            if (!n) continue;
            if (graph_ && mi.g) {
                mi.g->setEf(std::max<int>(ef_, k_));
                auto pq = mi.g->searchKnn(q, std::min<size_t>(k_, n));
                while (!pq.empty()) { all.push_back({pq.top().first, (i64)pq.top().second}); pq.pop(); }
            } else {
                for (size_t j = 0; j < n; ++j) all.push_back({l2sq(q, &mi.vecs[j * d_], d_), mi.ids[j]});
            }
        }
        size_t m = std::min<size_t>(k_, all.size());
        std::partial_sort(all.begin(), all.begin() + m, all.end());
        std::vector<i64> near(m);
        for (size_t j = 0; j < m; ++j) near[j] = all[j].second;
        double dk = m ? std::sqrt(std::max((double)all[m - 1].first, 0.0)) : 0.0;
        bool hit = false;
        if ((int)m >= k_) {
            auto it = theta_.find(region(q));
            hit = it != theta_.end() && dk <= (1.0 + D_) * it->second;
        }
        if (hit) for (i64 v : near) { auto o = owner_.find(v); if (o != owner_.end()) used_[o->second] = t; }
        return py::make_tuple(hit, near, dk);
    }

    // ids / vecs: the backend's top-k answer of the missed query x
    void on_miss(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                 py::array_t<i64, py::array::c_style | py::array::forcecast> ids,
                 py::array_t<float, py::array::c_style | py::array::forcecast> vecs, double t) {
        auto X = x.unchecked<1>(); auto I = ids.unchecked<1>(); auto V = vecs.unchecked<2>();
        ssize_t n = I.shape(0);
        if (!n) return;
        double dk = std::sqrt(std::max((double)l2sq(&X(0), &V(n - 1, 0), d_), 0.0));
        uint64_t key = region(&X(0));
        auto th = theta_.find(key);
        if (th == theta_.end()) theta_[key] = dk; else th->second = (1 - alpha_) * th->second + alpha_ * dk;
        std::vector<ssize_t> fresh;
        for (ssize_t j = 0; j < n; ++j) if (!owner_.count(I(j))) fresh.push_back(j);
        n_fetched_ += fresh.size();          // doc 9au: vectors QVCache fetches from the backend (re-reads)
        if (minis_[fill_].ids.size() + fresh.size() > cap_) next_fill();
        Mini& mi = minis_[fill_];
        if (graph_ && !mi.g) mi.g.reset(new hnswlib::HierarchicalNSW<float>(&space_, cap_ + k_ + 1, M_, efc_, 100));
        for (ssize_t j : fresh) {
            i64 v = I(j);
            if (owner_.count(v)) continue;           // duplicate id within the answer
            owner_[v] = fill_;
            mi.ids.push_back(v);
            mi.vecs.insert(mi.vecs.end(), &V(j, 0), &V(j, 0) + d_);
            if (graph_) mi.g->addPoint(&V(j, 0), (size_t)v);
        }
        used_[fill_] = t;
    }

    size_t n_cached() const { return owner_.size(); }
    size_t n_fetched() const { return n_fetched_; }
    int n_minis_used() const { int c = 0; for (auto& m : minis_) c += !m.ids.empty(); return c; }
    size_t n_regions() const { return theta_.size(); }
    std::vector<i64> cached_ids() const { std::vector<i64> o; for (auto& p : owner_) o.push_back(p.first); return o; }

private:
    size_t n_fetched_ = 0;
    struct Mini {
        std::vector<i64> ids;
        std::vector<float> vecs;
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> g;
    };

    void next_fill() {
        for (int i = 0; i < n_; ++i) if (minis_[i].ids.empty()) { fill_ = i; return; }
        int v = 0;
        for (int i = 1; i < n_; ++i) if (used_[i] < used_[v]) v = i;
        for (i64 id : minis_[v].ids) { auto o = owner_.find(id); if (o != owner_.end() && o->second == v) owner_.erase(o); }
        minis_[v].ids.clear(); minis_[v].vecs.clear(); minis_[v].g.reset();
        fill_ = v;
    }

    int d_, k_, n_;
    size_t cap_;
    double D_, alpha_;
    int nb_;
    bool graph_;
    int M_, efc_, ef_;
    int dims_ = 0;
    hnswlib::L2Space space_;
    std::vector<double> mu_, P_, lo_, hi_;
    std::vector<Mini> minis_;
    std::vector<double> used_;
    std::unordered_map<i64, int> owner_;
    std::unordered_map<uint64_t, double> theta_;
    int fill_ = 0;
};
