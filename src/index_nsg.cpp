#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <unordered_map>
#include <queue>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"

namespace efanna2e {
#define _CONTROL_NUM 100

    IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                       Index *initializer)
            : Index(dimension, n, m), initializer_{initializer} {}

    IndexNSG::~IndexNSG() {}

    void IndexNSG::Save(const char *filename) {
        std::ofstream out(filename, std::ios::binary | std::ios::out);
        assert(final_graph_.size() == nd_);

        out.write((char *) &width, sizeof(unsigned));
        out.write((char *) &ep_, sizeof(unsigned));
        for (unsigned i = 0; i < nd_; i++) {
            unsigned GK = (unsigned) final_graph_[i].size();
            out.write((char *) &GK, sizeof(unsigned));
            out.write((char *) final_graph_[i].data(), GK * sizeof(unsigned));
        }
        out.close();
    }

    void IndexNSG::Load(const char *filename) {
        std::ifstream in(filename, std::ios::binary);
        in.read((char *) &width, sizeof(unsigned));
        in.read((char *) &ep_, sizeof(unsigned));
        // width=100;
        unsigned cc = 0;
        while (!in.eof()) {
            unsigned k;
            in.read((char *) &k, sizeof(unsigned));
            if (in.eof()) break;
            cc += k;
            std::vector<unsigned> tmp(k);
            in.read((char *) tmp.data(), k * sizeof(unsigned));
            final_graph_.push_back(tmp);
        }
        cc /= nd_;
        // std::cout<<cc<<std::endl;
    }


    float IndexNSG::Test_neighbor_distance(const float *x) {
        data_ = x;
        float ave_dist = 0;
        for (unsigned id = 0; id < 100; id++) {
            float ans = -1;
            for (auto nxt:final_graph_[id]) {
                float dist = naive_dist_calc(data_ + dimension_ * id,data_ + dimension_ * nxt,dimension_);
                if (ans < 0) ans = dist;
                else ans = std::min(ans, dist);
            }
            std::cout << ans << " \n";
        }
    }


    void IndexNSG::Load_nn_graph(const char *filename) {
        std::ifstream in(filename, std::ios::binary);
        unsigned k;
        in.read((char *) &k, sizeof(unsigned));
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t) ss;
        size_t num = (unsigned) (fsize / (k + 1) / 4);
        in.seekg(0, std::ios::beg);

        final_graph_.resize(num);
        final_graph_.reserve(num);
        unsigned kk = (k + 3) / 4 * 4;
        for (size_t i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur);
            final_graph_[i].resize(k);
            final_graph_[i].reserve(kk);
            in.read((char *) final_graph_[i].data(), k * sizeof(unsigned));
        }
        in.close();
    }

    void IndexNSG::enhance() {
        int m = nd_;
        for (int i = 0; i < m; i++) {
            int nxt = (i + 1) % m;
            int pre = (i - 1 + m) % m;
            for (auto u:final_graph_[i]) {
                int check = u;
                if (check == nxt) nxt = -1;
                if (check == pre) pre = -1;
            }
            if (nxt != -1) final_graph_[i].push_back(nxt);
            if (pre != -1) final_graph_[i].push_back(pre);
        }
    }


    void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                                 std::vector<Neighbor> &retset,
                                 std::vector<Neighbor> &fullset) {
        unsigned L = parameter.Get<unsigned>("L");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

        boost::dynamic_bitset<> flags{nd_, 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
            init_ids[i] = final_graph_[ep_][i];
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            // std::cout<<id<<std::endl;
            float dist = distance_->compare(data_ + dimension_ * (size_t) id, query,
                                            (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            // flags[id] = 1;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = distance_->compare(query, data_ + dimension_ * (size_t) id,
                                                    (unsigned) dimension_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                                 boost::dynamic_bitset<> &flags,
                                 std::vector<Neighbor> &retset,
                                 std::vector<Neighbor> &fullset) {
        unsigned L = parameter.Get<unsigned>("L");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
            init_ids[i] = final_graph_[ep_][i];
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            // std::cout<<id<<std::endl;
            float dist = distance_->compare(data_ + dimension_ * (size_t) id, query,
                                            (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            fullset.push_back(retset[i]);
            // flags[id] = 1;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = distance_->compare(query, data_ + dimension_ * (size_t) id,
                                                    (unsigned) dimension_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    void IndexNSG::init_graph(const Parameters &parameters) {
        float *center = new float[dimension_];
        for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
        for (unsigned i = 0; i < nd_; i++) {
            for (unsigned j = 0; j < dimension_; j++) {
                center[j] += data_[i * dimension_ + j];
            }
        }
        for (unsigned j = 0; j < dimension_; j++) {
            center[j] /= nd_;
        }
        std::vector<Neighbor> tmp, pool;
        ep_ = rand() % nd_;  // random initialize navigating point
        get_neighbors(center, parameters, tmp, pool);
        ep_ = tmp[0].id;
    }

    void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                              const Parameters &parameter,
                              boost::dynamic_bitset<> &flags,
                              SimpleNeighbor *cut_graph_) {
        unsigned range = parameter.Get<unsigned>("R");
        unsigned maxc = parameter.Get<unsigned>("C");
        width = range;
        unsigned start = 0;

        for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
            unsigned id = final_graph_[q][nn];
            if (flags[id]) continue;
            float dist =
                    distance_->compare(data_ + dimension_ * (size_t) q,
                                       data_ + dimension_ * (size_t) id, (unsigned) dimension_);
            pool.push_back(Neighbor(id, dist, true));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Neighbor> result;
        if (pool[start].id == q) start++;
        result.push_back(pool[start]);

        while (result.size() < range && (++start) < pool.size() && start < maxc) {
            auto &p = pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < result.size(); t++) {
                if (p.id == result[t].id) {
                    occlude = true;
                    break;
                }
                float djk = distance_->compare(data_ + dimension_ * (size_t) result[t].id,
                                               data_ + dimension_ * (size_t) p.id,
                                               (unsigned) dimension_);
                if (djk < p.distance /* dik */) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(p);
        }

        SimpleNeighbor *des_pool = cut_graph_ + (size_t) q * (size_t) range;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void IndexNSG::InterInsert(unsigned n, unsigned range,
                               std::vector<std::mutex> &locks,
                               SimpleNeighbor *cut_graph_) {
        SimpleNeighbor *src_pool = cut_graph_ + (size_t) n * (size_t) range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1) break;

            SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            SimpleNeighbor *des_pool = cut_graph_ + des * (size_t) range;

            std::vector<SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                LockGuard guard(locks[des]);
                for (size_t j = 0; j < range; j++) {
                    if (des_pool[j].distance == -1) break;
                    if (n == des_pool[j].id) {
                        dup = 1;
                        break;
                    }
                    temp_pool.push_back(des_pool[j]);
                }
            }
            if (dup) continue;

            temp_pool.push_back(sn);
            if (temp_pool.size() > range) {
                std::vector<SimpleNeighbor> result;
                unsigned start = 0;
                std::sort(temp_pool.begin(), temp_pool.end());
                result.push_back(temp_pool[start]);
                while (result.size() < range && (++start) < temp_pool.size()) {
                    auto &p = temp_pool[start];
                    bool occlude = false;
                    for (unsigned t = 0; t < result.size(); t++) {
                        if (p.id == result[t].id) {
                            occlude = true;
                            break;
                        }
                        float djk = distance_->compare(data_ + dimension_ * (size_t) result[t].id,
                                                       data_ + dimension_ * (size_t) p.id,
                                                       (unsigned) dimension_);
                        if (djk < p.distance /* dik */) {
                            occlude = true;
                            break;
                        }
                    }
                    if (!occlude) result.push_back(p);
                }
                {
                    LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                }
            } else {
                LockGuard guard(locks[des]);
                for (unsigned t = 0; t < range; t++) {
                    if (des_pool[t].distance == -1) {
                        des_pool[t] = sn;
                        if (t + 1 < range) des_pool[t + 1].distance = -1;
                        break;
                    }
                }
            }
        }
    }

    void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
        /*
        std::cout << " graph link" << std::endl;
        unsigned progress=0;
        unsigned percent = 100;
        unsigned step_size = nd_/percent;
        std::mutex progress_lock;
        */
        unsigned range = parameters.Get<unsigned>("R");
        std::vector<std::mutex> locks(nd_);

#pragma omp parallel
        {
            // unsigned cnt = 0;
            std::vector<Neighbor> pool, tmp;
            boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < nd_; ++n) {
                pool.clear();
                tmp.clear();
                flags.reset();
                get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
                sync_prune(n, pool, parameters, flags, cut_graph_);
                /*
              cnt++;
              if(cnt % step_size == 0){
                LockGuard g(progress_lock);
                std::cout<<progress++ <<"/"<< percent << " completed" << std::endl;
                }
                */
            }
        }

#pragma omp for schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; ++n) {
            InterInsert(n, range, locks, cut_graph_);
        }
    }

    void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters) {
        std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
        unsigned range = parameters.Get<unsigned>("R");
        Load_nn_graph(nn_graph_path.c_str());
        data_ = data;
        init_graph(parameters);
        SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t) range];
        Link(parameters, cut_graph_);
        final_graph_.resize(nd_);

        for (size_t i = 0; i < nd_; i++) {
            SimpleNeighbor *pool = cut_graph_ + i * (size_t) range;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < range; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            final_graph_[i].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                final_graph_[i][j] = pool[j].id;
            }
        }

        tree_grow(parameters);

        unsigned max = 0, min = 1e6, avg = 0;
        for (size_t i = 0; i < nd_; i++) {
            auto size = final_graph_[i].size();
            max = max < size ? size : max;
            min = min > size ? size : min;
            avg += size;
        }
        avg /= 1.0 * nd_;
        printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

        has_built = true;
    }

    void IndexNSG::BFS() {
        std::queue<unsigned> Q;
        boost::dynamic_bitset<> flags{nd_, 0};
        Q.push(ep_);
        Q.push(1);
        flags[ep_] = 1;
        int max_deep = 0;
        float ave_deep = 0.00;
        while (!Q.empty()) {
            int u = Q.front();
            Q.pop();
            int deep = Q.front();
            Q.pop();
            ave_deep += deep;
            max_deep = std::max(deep, max_deep);
            for (auto v:final_graph_[u]) {
                if (flags[v]) continue;
                else {
                    flags[v] = 1;
                    Q.push(v);
                    Q.push(deep + 1);
                }
            }
        }
        std::cout << max_deep << " " << ave_deep / float(nd_) << " \n";
    }

    float IndexNSG::naive_dist_calc(const float *q, const float *p, const unsigned &dim) {
        float ans = 0.0;
        for (unsigned i = 0; i < dim; i++) {
            ans += (p[i] - q[i]) * (p[i] - q[i]);
        }
        return ans;
    }

    float IndexNSG::product_dist_calc(const unsigned id, const float *q, const unsigned int &dim,
                                      const unsigned int &sub_dim) {
        //printf("porduct id %d\n",id);
        float ans = 0.0;
        for (unsigned i = 0, sub_id = 0; i < dim; i += sub_dim, sub_id++) {
            int cluster_id = quant_vector[id][sub_id];
            for (int j = 0; j < sub_dim; j++) {
                ans += (code_vec[sub_id][cluster_id][j] - q[i + j]) * (code_vec[sub_id][cluster_id][j] - q[i + j]);
            }
        }
        //printf("test id dist is %f\n",ans);
        return ans;
    }

    float IndexNSG::product_hybrid_calc(const unsigned id, const float *q, const unsigned int &dim) {
        //printf("porduct id %d\n",id);
        float ans = 0.0;
        sub_dim = 128 / quant_vector[id].size();
        int codebook_id = -1;
        if (sub_dim == 1) codebook_id = 2;
        else if (sub_dim == 2) codebook_id = 1;
        else if (sub_dim == 4) codebook_id = 0;
        for (unsigned i = 0, sub_id = 0; i < dim; i += sub_dim, sub_id++) {
            int cluster_id = quant_vector[id][sub_id];
            for (int j = 0; j < sub_dim; j++) {
                ans += (all_code_vec[codebook_id][sub_id][cluster_id][j] - q[i + j]) *
                       (all_code_vec[codebook_id][sub_id][cluster_id][j] - q[i + j]);
            }
        }
        //printf("test id dist is %f\n",ans);
        return ans;
    }

    float IndexNSG::progress_table_dist(const unsigned id, const float *q, const unsigned int &dim,
                                        const unsigned int &sub_dim, boost::dynamic_bitset<> &progress_bitset) {
        //printf("progress id %d\n",id);
        float ans = 0.0;
        for (unsigned i = 0, sub_id = 0; i < dim; i += sub_dim, sub_id++) {
            int map_id = cluster_num * sub_id + quant_vector[id][sub_id];
            if (progress_bitset[map_id])
                ans += progress_hash_table[map_id];
            else {
                float res = 0;
                int cluster_id = quant_vector[id][sub_id];
                for (unsigned j = 0; j < sub_dim; j++) {
                    res += (code_vec[sub_id][cluster_id][j] - q[i + j]) * (code_vec[sub_id][cluster_id][j] - q[i + j]);
                }
                ans += res;
                progress_hash_table[map_id] = res;
                progress_bitset[map_id] = true;
            }
        }
        //printf("test id dist is %f\n",ans);
        return ans;
    }

    float IndexNSG::full_table_dist(const unsigned id, const float *q, const unsigned int &dim,
                                    const unsigned int &sub_dim) {
        float ans = 0.0;
        for (unsigned i = 0, sub_id = 0; i < dim; i += sub_dim, sub_id++) {
            int map_id = cluster_num * sub_id + quant_vector[id][sub_id];
            ans += full_hash_table[map_id];
        }
        return ans;
    }

    void IndexNSG::Search(const float *query, const float *x, size_t K,
                          const Parameters &parameters, unsigned *indices) {
        const unsigned L = parameters.Get<unsigned>("L_search");
        data_ = x;
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
            init_ids[tmp_l] = final_graph_[ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist = naive_dist_calc(data_ + dimension_ * id, query, (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float dist = naive_dist_calc(data_ + dimension_ * id, query, (unsigned) dimension_);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::Product_Search(const float *query, size_t K,
                                  const Parameters &parameters, unsigned *indices) {
        const unsigned L = parameters.Get<unsigned>("L_search");

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
            init_ids[tmp_l] = final_graph_[ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist = product_dist_calc(id, query, dimension_, sub_dim);
            retset[i] = Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float dist = product_dist_calc(id, query, dimension_, sub_dim);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::Product_Hybrid_Search(const float *query, size_t K,
                                         const Parameters &parameters, unsigned *indices) {
        const unsigned L = parameters.Get<unsigned>("L_search");

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
            init_ids[tmp_l] = final_graph_[ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            sub_dim = 128 / quant_vector[id].size();
            float dist = product_hybrid_calc(id, query, dimension_);
            retset[i] = Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    sub_dim = 128 / quant_vector[id].size();
                    float dist = product_hybrid_calc(id, query, dimension_);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }


    void IndexNSG::Product_Progress_Search(const float *query, size_t K,
                                           const Parameters &parameters, unsigned *indices) {
        const unsigned L = parameters.Get<unsigned>("L_search");
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        boost::dynamic_bitset<> progress_bitset{sub_count * cluster_num, 0};
        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
            init_ids[tmp_l] = final_graph_[ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist = progress_table_dist(id, query, dimension_, sub_dim, progress_bitset);
            retset[i] = Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float dist = progress_table_dist(id, query, dimension_, sub_dim, progress_bitset);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::Product_Table_Search(const float *query, size_t K,
                                        const Parameters &parameters, unsigned *indices) {
        const unsigned L = parameters.Get<unsigned>("L_search");
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
        for (unsigned i = 0, sub_id = 0; i < dimension_; i += sub_dim, sub_id++) {
            for (unsigned j = 0; j < cluster_num; j++) {
                int map_id = sub_id * cluster_num + j;
                float res = 0.0;
                for (unsigned k = 0; k < sub_dim; k++) {
                    res += (query[i + k] - code_vec[sub_id][j][k]) * (query[i + k] - code_vec[sub_id][j][k]);
                }
                full_hash_table[map_id] = res;
            }
        }

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
            init_ids[tmp_l] = final_graph_[ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist = full_table_dist(id, query, dimension_, sub_dim);
            retset[i] = Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float dist = full_table_dist(id, query, dimension_, sub_dim);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::Single_Search(const float *query, const float *x, size_t K,
                                 const Parameters &parameters, unsigned *indices) {
        const unsigned L = parameters.Get<unsigned>("L_search");
        data_ = x;
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        boost::dynamic_bitset<> inqueue{nd_, 0};
        std::unordered_map<unsigned, float> mp;
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
        unsigned count = L;
        unsigned cur = ep_;
        float ans_dist = distance_->compare(data_ + dimension_ * cur, query, (unsigned) dimension_);
        indices[0] = cur;
        flags[cur] = 1;
        while (count--) {
            inqueue[cur] = 1;
            float dist = 1e9;
            unsigned candidate = nd_;
            for (unsigned &id : final_graph_[cur]) {
                if (inqueue[id]) continue;
                if (!flags[id]) {
                    flags[id] = 1;
                    mp[id] = distance_->compare(data_ + dimension_ * id, query, (unsigned) dimension_);
                    if (mp[id] < ans_dist) indices[0] = id, ans_dist = mp[id];
                }
                if (candidate == nd_ || mp[id] < dist) {
                    dist = mp[id];
                    candidate = id;
                }
            }
            cur = candidate;
        }
    }


    void IndexNSG::Energy_Calc(float *energy) {

    }

    void IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                      const Parameters &parameters, unsigned *indices) {
        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            //_mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                //_mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
//      for (unsigned m = 0; m < MaxM; ++m)
//        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::OptimizeGraph(float *data) {  // use after build or load

        data_ = data;
        data_len = (dimension_ + 1) * sizeof(float);
        neighbor_len = (width + 1) * sizeof(unsigned);
        node_size = data_len + neighbor_len;
        opt_graph_ = (char *) malloc(node_size * nd_);
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        for (unsigned i = 0; i < nd_; i++) {
            char *cur_node_offset = opt_graph_ + i * node_size;
            float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
            std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
            std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                        data_len - sizeof(float));

            cur_node_offset += data_len;
            unsigned k = final_graph_[i].size();
            std::memcpy(cur_node_offset, &k, sizeof(unsigned));
            std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                        k * sizeof(unsigned));
            std::vector<unsigned>().swap(final_graph_[i]);
        }
        CompactGraph().swap(final_graph_);
    }

    void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root]) cnt++;
        flag[root] = true;
        while (!s.empty()) {
            unsigned next = nd_ + 1;
            for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
                if (flag[final_graph_[tmp][i]] == false) {
                    next = final_graph_[tmp][i];
                    break;
                }
            }
            // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
            if (next == (nd_ + 1)) {
                s.pop();
                if (s.empty()) break;
                tmp = s.top();
                continue;
            }
            tmp = next;
            flag[tmp] = true;
            s.push(tmp);
            cnt++;
        }
    }

    void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                            const Parameters &parameter) {
        unsigned id = nd_;
        for (unsigned i = 0; i < nd_; i++) {
            if (flag[i] == false) {
                id = i;
                break;
            }
        }

        if (id == nd_) return;  // No Unlinked Node

        std::vector<Neighbor> tmp, pool;
        get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
        std::sort(pool.begin(), pool.end());

        unsigned found = 0;
        for (unsigned i = 0; i < pool.size(); i++) {
            if (flag[pool[i].id]) {
                // std::cout << pool[i].id << '\n';
                root = pool[i].id;
                found = 1;
                break;
            }
        }
        if (found == 0) {
            while (true) {
                unsigned rid = rand() % nd_;
                if (flag[rid]) {
                    root = rid;
                    break;
                }
            }
        }
        final_graph_[root].push_back(id);
    }

    void IndexNSG::tree_grow(const Parameters &parameter) {
        unsigned root = ep_;
        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned unlinked_cnt = 0;
        while (unlinked_cnt < nd_) {
            DFS(flags, root, unlinked_cnt);
            // std::cout << unlinked_cnt << '\n';
            if (unlinked_cnt >= nd_) break;
            findroot(flags, root, parameter);
            // std::cout << "new root"<<":"<<root << '\n';
        }
        for (size_t i = 0; i < nd_; ++i) {
            if (final_graph_[i].size() > width) {
                width = final_graph_[i].size();
            }
        }
    }
}
