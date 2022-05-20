
#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <unordered_map>

std::unordered_map<int, int> mp;

void load_data(char *filename, float *&data, unsigned &num,
               unsigned &dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}

void load_data_int(char *filename, int *&data, unsigned &num,
               unsigned &dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    data = new int[num * dim * sizeof(int)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}

void save_result(char *filename, std::vector<std::vector<unsigned> > &results) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for (unsigned i = 0; i < results.size(); i++) {
        unsigned GK = (unsigned) results[i].size();
        out.write((char *) &GK, sizeof(unsigned));
        out.write((char *) results[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}

double test_result(std::vector<std::vector<unsigned> > &results, int *&gt, int query_num, int k) {
    double base = 0.00, check = 0.00;
    base = query_num * k;
    for (int i = 0; i < query_num; i++) {
        mp.clear();
        for (int j = 0; j < k; j++) {
            int u = *(gt + i * 100 + j);
            mp[u] = 1;
        }
        for (int j = 0; j < k; j++) {
            if (mp[results[i][j]]) check += 1.0;
        }
    }
    return check / base;
}


int main(int argc, char **argv) {
    if (argc != 8) {
        std::cout << argv[0]
                  << " data_file query_file nsg_path search_L search_K result_path ground_truth"
                  << std::endl;
        exit(-1);
    }
    float *data_load = NULL;
    unsigned points_num, dim;
    load_data(argv[1], data_load, points_num, dim);
    float *query_load = NULL;
    float *node_energy = NULL;
    unsigned query_num, query_dim;
    load_data(argv[2], query_load, query_num, query_dim);

    assert(dim == query_dim);
    int *ground_load = NULL;
    unsigned ground_num, ground_dim;
    load_data_int(argv[7], ground_load, ground_num, ground_dim);

    node_energy = new float[points_num * sizeof(float)];
    unsigned L = (unsigned) atoi(argv[4]);
    unsigned K = (unsigned) atoi(argv[5]);

    if (L < K) {
        std::cout << "search_L cannot be smaller than search_K!" << std::endl;
        exit(-1);
    }

    // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
    // align the data before build query_load = efanna2e::data_align(query_load,
    // query_num, query_dim);
    efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
    index.Load(argv[3]);
    //index.enhance();
    index.BFS();
    index.Energy_Calc(node_energy);
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    auto s = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<unsigned> > res;
    std::cout << "query num " << query_num << " \n";
    for (unsigned i = 0; i < query_num; i++) {
        std::vector<unsigned> tmp(K);
        index.Search(query_load + i * dim, data_load, K, paras, tmp.data());
        res.push_back(tmp);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    double time_slap = diff.count();
    std::cout << "search time: " << time_slap << std::endl;
    std::cout << "query per s: " << query_num * 1.0 / time_slap << std::endl;
    std::cout << test_result(res, ground_load, query_num, K) << std::endl;
    save_result(argv[6], res);

    return 0;
}
