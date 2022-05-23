//
// Created by Lenovo on 2022/5/16.
//
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cmath>

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <queue>

std::vector<std::vector<std::vector<float> > > code_vec, all_code_vec[3];
std::vector<std::vector<unsigned char> > quant_vector;

struct quant_item {
    int id;
    int quant_mod;
    float loss, value;

    bool operator<(const quant_item &a) const {
        return value < a.value;
    }
};

std::priority_queue<quant_item> quant_queue;

bool isFileExists_ifstream(char *filename) {
    std::ifstream f(filename);
    return f.good();
}

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

float dist_calc(float *&data, int dim, int p, int q) {
    float ans = 0;
    for (int i = 0; i < dim; i++) {
        ans += (data[p * dim + i] - data[q * dim + i]) * (data[p * dim + i] - data[q * dim + i]);
    }
    return ans;
}

float dist_vec_cluster(float *&data, int dim, int p, std::vector<float> &q) {
    float ans = 0;
    for (int i = 0; i < dim; i++) {
        ans += (data[p * dim + i] - q[i]) * (data[p * dim + i] - q[i]);
    }
    return ans;
}

float dist_vec(const float *p, int dim, std::vector<float> &q) {
    float ans = 0;
    for (int i = 0; i < dim; i++) {
        ans += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return ans;
}


float dist_cluster_cluster(float *&data, int dim, std::vector<float> &p, std::vector<float> &q) {
    float ans = 0;
    for (int i = 0; i < dim; i++) {
        ans += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return ans;
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
    //932085
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
    std::unordered_map<int, int> mp;
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
    std::unordered_map<int, int>().swap(mp);
    return check / base;
}

void Save_code_vec(const char *filename) {
    freopen(filename, "w", stdout);
    printf("%d %d %d\n", code_vec.size(), code_vec[0].size(), code_vec[0][0].size());
    for (auto &u:code_vec) {
        for (auto &v:u) {
            for (auto &x:v) {
                printf("%.6f ", x);
            }
            puts("");
        }
        puts("");
    }
    fclose(stdout);
}

void Load_code_book(const char *filename) {
    freopen(filename, "r", stdin);
    int x, y, z;
    scanf("%d %d %d", &x, &y, &z);
    code_vec.resize(x);
    for (int i = 0; i < x; i++) {
        code_vec[i].resize(y);
        for (int j = 0; j < y; j++) {
            code_vec[i][j].resize(z);
            for (int k = 0; k < z; k++) {
                scanf("%f", &code_vec[i][j][k]);
            }
        }
    }
    fclose(stdin);
}

void Load_code_book_ifsteam(const char *filename) {
    std::ifstream in(filename);
    int x, y, z;
    in >> x >> y >> z;
    code_vec.resize(x);
    for (int i = 0; i < x; i++) {
        code_vec[i].resize(y);
        for (int j = 0; j < y; j++) {
            code_vec[i][j].resize(z);
            for (int k = 0; k < z; k++) {
                in >> code_vec[i][j][k];
            }
        }
    }
    in.close();
}

void Save_quantization_data(const char *filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    unsigned sub_count = quant_vector.size();
    unsigned sub_dim = quant_vector[0].size();
    out.write((char *) &sub_count, sizeof(unsigned));
    out.write((char *) &sub_dim, sizeof(unsigned));
    for (unsigned i = 0; i < sub_count; i++) {
        for (unsigned j = 0; j < sub_dim; j++) {
            out.write((char *) &quant_vector[i][j], sizeof(unsigned char));
        }
    }
    out.close();
}

void Load_quantization_data(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    unsigned sub_count, sub_dim;
    in.read((char *) &sub_count, sizeof(unsigned));
    in.read((char *) &sub_dim, sizeof(unsigned));
    quant_vector.resize(sub_count);
    for (unsigned i = 0; i < sub_count; i++) {
        quant_vector[i].resize(sub_dim);
        for (unsigned j = 0; j < sub_dim; j++) {
            in.read((char *) &quant_vector[i][j], sizeof(unsigned char));
        }
    }
    in.close();
}


void generate_K_means(float *&data, int dim, int train_num, int K, int iter_count, int code_count) {
    std::vector<std::vector<float> > cluster;
    std::vector<int> node_cluster;
    std::vector<int> num_in_cluster;
    cluster.resize(K);
    num_in_cluster.resize(K);
    node_cluster.resize(train_num);
    for (int i = 0; i < K; i++) {
        unsigned init_id = rand() % train_num;
        cluster[i].resize(dim);
        for (int j = 0; j < dim; j++) {
            cluster[i][j] = data[init_id * dim + j];
        }
    }
    printf("code_count %d\n", code_count);
    while (iter_count--) {
        double ave_dist = 0.00, base = 0.00, pre_loss = 0.00;
        if (iter_count % 10 == 0) printf(" iter time %d\n", iter_count);
        for (int i = 0; i < train_num; i++) {
            int belong_to = -1;
            float dist = 0.00;
            for (int j = 0; j < K; j++) {
                float new_dist = dist_vec_cluster(data, dim, i, cluster[j]);
                if (belong_to == -1) {
                    belong_to = j;
                    dist = new_dist;
                } else if (dist > new_dist) {
                    belong_to = j;
                    dist = new_dist;
                }
            }
            ave_dist += dist;
            base += 1.0;
            //std::cout<<i<<" "<<belong_to<<"\n";
            node_cluster[i] = belong_to;
        }

        if (iter_count % 10 == 0) printf("loss :: %.6f\n", ave_dist / base);
        pre_loss = ave_dist / base;

        for (int i = 0; i < K; i++) {
            num_in_cluster[i] = 0;
            for (int j = 0; j < dim; j++) {
                cluster[i][j] = 0;
            }
        }
        for (int i = 0; i < train_num; i++) {
            int belong_to = node_cluster[i];
            num_in_cluster[belong_to]++;
            for (int j = 0; j < dim; j++) {
                cluster[belong_to][j] += data[i * dim + j];
            }
        }
        for (int i = 0; i < K; i++) {
            if (num_in_cluster[i] == 0) {
                continue;
            }
            for (int j = 0; j < dim; j++) {
                cluster[i][j] /= (float) num_in_cluster[i];
            }
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < dim; j++) {
            code_vec[code_count][i][j] = cluster[i][j];
        }
    }
    std::vector<std::vector<float> >().swap(cluster);
    std::vector<int>().swap(node_cluster);
    std::vector<int>().swap(num_in_cluster);
    return;
}

float quant_item_loss_calc(float *data_item, int dim, int sub_dim, int K) {
    int codebook_id = -1;
    if (sub_dim == 1) codebook_id = 2;
    else if (sub_dim == 2) codebook_id = 1;
    else if (sub_dim == 4) codebook_id = 0;
    float all_dist = 0;
    for (int i = 0, sub_id = 0; i < dim; i += sub_dim, sub_id++) {
        int belong = -1;
        float dist = 0.00;
        for (int k = 0; k < K; k++) {
            float new_dist = dist_vec(data_item, sub_dim, all_code_vec[codebook_id][sub_id][k]);
            if (belong == -1) {
                belong = k;
                dist = new_dist;
            } else if (new_dist < dist) {
                belong = k;
                dist = new_dist;
            }
        }
        all_dist += dist;
    }
    return sqrt(all_dist);
}

float quant_item_promote(float *data_item, unsigned id, unsigned dim, unsigned sub_dim, int K) {
    int codebook_id = -1;
    if (sub_dim == 1) codebook_id = 2;
    else if (sub_dim == 2) codebook_id = 1;
    else if (sub_dim == 4) codebook_id = 0;
    float all_dist = 0;
    //std::cout<<sub_dim<<" \n";
    quant_vector[id].resize(dim / sub_dim);
    for (unsigned i = 0, sub_id = 0; i < dim; i += sub_dim, sub_id++) {
        int belong = -1;
        float dist = 0.00;
        for (unsigned k = 0; k < K; k++) {
            //std::cout<<codebook_id<<" "<<sub_id<<" "<<k<<" \n";
            float new_dist = dist_vec(data_item, sub_dim, all_code_vec[codebook_id][sub_id][k]);
            if (belong == -1) {
                belong = k;
                dist = new_dist;
            } else if (new_dist < dist) {
                belong = k;
                dist = new_dist;
            }
        }
        all_dist += dist;
        quant_vector[id][sub_id] = belong;
    }
    return sqrt(all_dist);
}


void generate_product_quantization(float *data, int num, int dim, int sub_dim, int K) {
    int sub_count = dim / sub_dim;
    for (int i = 0, sub_id = 0; i < dim; i += sub_dim, sub_id++) {
        printf("now finished %d\n", sub_id);
        double ave_dist = 0, base = 0;
        for (int j = 0; j < num; j++) {
            int belong = -1;
            float dist = 0.00;
            for (int k = 0; k < K; k++) {
                float new_dist = dist_vec(&data[j * dim + i], sub_dim, code_vec[sub_id][k]);
                if (belong == -1) {
                    belong = k;
                    dist = new_dist;
                } else if (new_dist < dist) {
                    belong = k;
                    dist = new_dist;
                }
            }
            ave_dist += dist;
            base += 1.0;
            quant_vector[j][sub_id] = belong;
        }
    }
}


int main(int argc, char **argv) {
    srand(0);
    float *train_data = NULL;
    unsigned train_num, dim;
    load_data(argv[1], train_data, train_num, dim);
    std::cout << "tarin item count:: " << train_num << std::endl;
    unsigned sub_count = (unsigned) atoi(argv[2]);
    unsigned cluster_count = (unsigned) atoi(argv[3]);
    unsigned sub_len = dim / sub_count;
    unsigned sub_dim = sub_len;
    int require_size = 64000000;
    std::cout << sub_len << " " << cluster_count << std::endl;
    if (isFileExists_ifstream("code_vec32.nsg")) {
        Load_code_book_ifsteam("code_vec32.nsg");
        all_code_vec[0] = code_vec;
    }
    if (isFileExists_ifstream("code_vec64.nsg")) {
        Load_code_book_ifsteam("code_vec64.nsg");
        all_code_vec[1] = code_vec;
    }
    if (isFileExists_ifstream("code_vec128.nsg")) {
        Load_code_book_ifsteam("code_vec128.nsg");
        all_code_vec[2] = code_vec;
    }
    unsigned data_num;
    float *full_data;
    load_data(argv[4], full_data, data_num, dim);
//    for (int i = 0; i < 10; i++) {
//        for (int j = 0; j < dim; j++) {
//            printf("%.0f ", full_data[i * dim + j]);
//        }
//    }
//    0 16 35 5 32 31 14 10

    quant_vector.resize(data_num);
    int all_size = 0;
    for (int i = 0; i < data_num; i++) {
        quant_item u;
        if (i % 10000 == 0) {
            printf("finished %d\n", i);
        }

        u.id = i;
        u.loss = quant_item_promote(full_data + i*dim, i, dim, 4, cluster_count);
        u.quant_mod = 0;
        u.value = u.loss - quant_item_loss_calc(full_data + i, dim, 2, cluster_count);
        quant_queue.push(u);
        all_size += 32;
    }
    printf("all szie %d\n",all_size);
    while (all_size < require_size) {
        quant_item u = quant_queue.top();
        quant_queue.pop();
        if (u.value < 0) break;
        if (u.quant_mod == 2) break;
        all_size -= (32 * (1 << u.quant_mod));
        u.quant_mod += 1;
        all_size += (32 * (1 << u.quant_mod));
        int next_sub_dim = dim / (32 * (1 << u.quant_mod));
        u.loss = quant_item_promote(full_data + u.id*dim, u.id, dim, next_sub_dim, cluster_count);
        next_sub_dim = dim / (32 * (1 << u.quant_mod));
        if (u.quant_mod == 2) u.value = 0;
        else u.value = u.loss - quant_item_loss_calc(full_data + u.id, dim, next_sub_dim, cluster_count);
    }

    int cnt0 = 0, cnt1 = 0, cnt2 = 0;
    for (int i = 0; i < data_num; i++) {
        if (quant_vector[i].size() == 32) cnt0++;
        else if (quant_vector[i].size() == 64) cnt1++;
        else cnt2++;
    }
    printf(" size count %d %d %d\n", cnt0, cnt1, cnt2);



    // TEST loss
    double ans = 0.00, base = 0.00;
    for (unsigned i = 0; i < data_num; i++) {
        unsigned codebook_id = -1;
        if (quant_vector[i].size() == 32) codebook_id = 0, sub_len = 4;
        else if (quant_vector[i].size() == 64) codebook_id = 1, sub_len = 2;
        else if (quant_vector[i].size() == 128) codebook_id = 2, sub_len = 1;

        for (unsigned j = 0, sub_id = 0; j < dim; j += sub_len, sub_id++) {
            int map_id = quant_vector[i][sub_id];
            for (int k = 0; k < sub_len; k++) {
                int data_id = i * dim + j + k;
                ans += (all_code_vec[codebook_id][sub_id][map_id][k] - full_data[data_id]) *
                       (all_code_vec[codebook_id][sub_id][map_id][k] - full_data[data_id]);

                //printf("%f %f\n", code_vec[sub_id][map_id][k], full_data[data_id]);
            }
        }
        base += 1.0;
    }
    printf("loss is %.6f\n", ans / base);
    //128 loss is 92.733989
    //64 loss is 670.847607
    //32 loss is 3929.065150
    // 315962
    efanna2e::IndexNSG index(dim, data_num, efanna2e::L2, nullptr);
    index.Load(argv[5]);
    efanna2e::Parameters paras;
    unsigned L = (unsigned) atoi(argv[6]);
    unsigned K = (unsigned) atoi(argv[7]);
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    float *query_load = NULL;
    unsigned query_num, query_dim;
    load_data(argv[8], query_load, query_num, query_dim);

    int *ground_load = NULL;
    unsigned ground_num, ground_dim;
    load_data_int(argv[9], ground_load, ground_num, ground_dim);


    index.full_hash_table.resize(cluster_count * sub_count);
    index.progress_hash_table.resize(cluster_count * sub_count);
    index.code_vec.swap(code_vec);
    //index.quant_vector.swap(quant_vector);
    index.cluster_num = cluster_count;
    index.sub_dim = sub_len;
    index.sub_count = sub_count;
    std::cout << "Search Module:: Product Search" << std::endl;
    std::vector<std::vector<unsigned> > res;
    std::cout << "query num " << query_num << " \n";
    printf("K:: %d\n", K);
    for (L = 100; L <= 401; L += 25) {
        //std::cout << " K:: " << K << std::endl;
        //std::cout << " L:: " << L << std::endl;
        paras.Set<unsigned>("L_search", L);
        paras.Set<unsigned>("P_search", L);
        //printf("Product Search\n");
        res.clear();
        auto s = std::chrono::high_resolution_clock::now();
        for (unsigned i = 0; i < query_num; i++) {
            std::vector<unsigned> tmp(K);
            index.Search(query_load + i * dim, full_data, K, paras, tmp.data());
            //index.Product_Search(query_load + i * dim, K, paras, tmp.data());
            //index.Product_Progress_Search(query_load + i * dim, K, paras, tmp.data());
            //index.Product_Table_Search(query_load + i * dim, K, paras, tmp.data());
            res.push_back(tmp);
        }

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double time_slap = diff.count();

        float recall = test_result(res, ground_load, query_num, K);
        float Qms = query_num / time_slap / 1000.00;
        printf("%d,%f,%f\n", K, recall, Qms);

//        printf("Progress Search\n");
//        res.clear();
//        s = std::chrono::high_resolution_clock::now();
//        for (unsigned i = 0; i < query_num; i++) {
//            std::vector<unsigned> tmp(K);
//            //index.Search(query_load + i * dim, full_data, K, paras, tmp.data());
//            //index.Product_Search(query_load + i * dim, K, paras, tmp.data());
//            index.Product_Progress_Search(query_load + i * dim, K, paras, tmp.data());
//            //index.Product_Table_Search(query_load + i * dim, K, paras, tmp.data());
//            res.push_back(tmp);
//        }
//
//        e = std::chrono::high_resolution_clock::now();
//        diff = e - s;
//        time_slap = diff.count();
//        std::cout << "search time: " << time_slap << std::endl;
//        std::cout << "query per s: " << query_num * 1.0 / time_slap << std::endl;
//        std::cout << test_result(res, ground_load, query_num, K) << std::endl;
//
//        printf("Tabel Search\n");
//        res.clear();
//        s = std::chrono::high_resolution_clock::now();
//        for (unsigned i = 0; i < query_num; i++) {
//            std::vector<unsigned> tmp(K);
//            //index.Search(query_load + i * dim, full_data, K, paras, tmp.data());
//            //index.Product_Search(query_load + i * dim, K, paras, tmp.data());
//            //index.Product_Progress_Search(query_load + i * dim, K, paras, tmp.data());
//            index.Product_Table_Search(query_load + i * dim, K, paras, tmp.data());
//            res.push_back(tmp);
//        }
//        e = std::chrono::high_resolution_clock::now();
//        diff = e - s;
//        time_slap = diff.count();
//        std::cout << "search time: " << time_slap << std::endl;
//        std::cout << "query per s: " << query_num * 1.0 / time_slap << std::endl;
//        std::cout << test_result(res, ground_load, query_num, K) << std::endl;
    }

    return 0;
}
