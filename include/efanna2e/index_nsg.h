#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>

namespace efanna2e {

class IndexNSG : public Index {
 public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexNSG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;

    void Product_Search(
            const float *query,
            size_t k,
            const Parameters &parameters,
            unsigned *indices);
    void Product_Table_Search(
            const float *query,
            size_t k,
            const Parameters &parameters,
            unsigned *indices);
    void Product_Progress_Search(
            const float *query,
            size_t k,
            const Parameters &parameters,
            unsigned *indices);

    void Product_Hybrid_Search(
            const float *query,
            size_t k,
            const Parameters &parameters,
            unsigned *indices);

  void Single_Search(
        const float *query,
        const float *x,
        size_t k,
        const Parameters &parameters,
        unsigned *indices);
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices);
  void OptimizeGraph(float* data);
  float naive_dist_calc(const float *q,const float *p,const unsigned &dim);
  float full_table_dist(const unsigned id,const float *q,const unsigned &dim,const unsigned &sub_dim);
  float progress_table_dist(const unsigned id,const float *q,const unsigned &dim,const unsigned &sub_dim,boost::dynamic_bitset<> &progress_bitset);
  float product_dist_calc(const unsigned id,const float *q,const unsigned &dim,const unsigned &sub_dim);
  float product_hybrid_calc(const unsigned id, const float *q, const unsigned int &dim);
    void enhance();
  void BFS();
  void Energy_Calc(float* energy);

    std::vector<float> full_hash_table;
    std::vector<float> progress_hash_table;
    std::vector<std::vector<std::vector<float> > > code_vec;
    std::vector<std::vector<std::vector<float> > > all_code_vec[3];
    std::vector<std::vector<short> > quant_vector;
    unsigned cluster_num = 256;
    unsigned sub_dim = 2;
    unsigned sub_count = 64;
protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_;
    float* energy;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;
};
}

#endif //EFANNA2E_INDEX_NSG_H
