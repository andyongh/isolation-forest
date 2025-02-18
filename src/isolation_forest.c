
/*
High-performance anomaly detection algorithm - Isolation Forest implemented in C.

The MIT License (MIT)

Copyright (c) 2025 AndY

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "isolation_forest.h"

#include "ndarray.h"

struct itree_node {
    int split_feature;         // Index of feature used for splitting
    double split_value;        // Threshold value for splitting
    struct itree_node* left;   // Left subtree
    struct itree_node* right;  // Right subtree
    int sample_size;           // Number of samples in node
};

struct isolation_forest {
    itree_node** trees;  // Array of tree pointers
    int num_trees;       // Total number of trees
    int num_samples;     // Subsampling size per tree
    int max_depth;       // Maximum tree depth
    int num_threads;     // Number of parallel threads
    int num_features;    // Feature dimension
    double contamination;
    uint32_t random_state;
};

typedef struct {
    isolation_forest* forest;
    ndarray_t* data;
    int start_tree;
    int end_tree;
} thread_param;

// Recursively create tree node
static itree_node* create_node(double** data, int n_features, int start, int end, int depth, int max_depth)
{
    // int seed         = 42;
    itree_node* node = malloc(sizeof(itree_node));
    if (node == NULL) {
        return NULL;
    }
    node->left = node->right = NULL;

    // Termination conditions
    if (depth >= max_depth || end - start <= 1) {
        node->split_feature = -1;
        node->sample_size   = end - start;
        return node;
    }

    // Random feature selection
    // int feat_idx = rand_r(&seed) % n_features;
    int feat_idx = rand() % n_features;
    double min   = data[start][feat_idx];
    double max   = min;

    // Calculate feature range
    for (int i = start + 1; i < end; i++) {
        double val = data[i][feat_idx];
        if (val < min) min = val;
        if (val > max) max = val;
    }

    // Generate split value and partition data
    // double split_val = min + (max - min) * (rand_r(&seed) / (double)RAND_MAX);
    double split_val = min + (max - min) * (rand() / (double)RAND_MAX);
    int pivot        = start;
    for (int i = start; i < end; i++) {
        if (data[i][feat_idx] < split_val) {
            double* tmp = data[pivot];
            data[pivot] = data[i];
            data[i]     = tmp;
            pivot++;
        }
    }

    // Build subtrees recursively
    node->split_feature = feat_idx;
    node->split_value   = split_val;
    node->sample_size   = end - start;
    node->left          = create_node(data, n_features, start, pivot, depth + 1, max_depth);
    node->right         = create_node(data, n_features, pivot, end, depth + 1, max_depth);
    return node;
}

// Calculate path length to isolate data point
static int itree_get_path_len(itree_node* node, double* x)
{
    int len             = 0;
    itree_node* current = node;
    while (current != NULL && current->split_feature != -1) {
        if (x[current->split_feature] < current->split_value) {
            current = current->left;
        } else {
            current = current->right;
        }
        len++;
    }
    return len;
}

static void free_tree(itree_node* node)
{
    if (node) {
        free_tree(node->left);
        free_tree(node->right);
        free(node);
    }
}

static double** ndarray_sample_without_replacement(ndarray_t* data, uint64_t* sample_size)
{
    // int seed       = 42;
    uint64_t total = data->dimensions[0];
    *sample_size   = (*sample_size < total) ? *sample_size : total;
    // printf("sample_size[%llu] total[%llu].\n", *sample_size, total);
    double** result = (double**)calloc(*sample_size, sizeof(double*));

    if (!result) {
        printf("Memory allocation failed.\n");
        return NULL;
    }

    double** temp = (double**)calloc(total, sizeof(double*));
    if (!temp) {
        printf("Memory allocation failed.\n");
        free(result);
        return NULL;
    }

    for (uint64_t i = 0; i < total; i++) {
        uint64_t stride = data->strides[0];
        temp[i]         = &data->data[i * stride];
    }

    // Fisher-Yates Shuffle
    for (uint64_t i = 0; i < *sample_size; i++) {
        // int j = i + rand_r(&seed) % (*sample_size - i);
        int j = i + rand() % (total - i);
        // printf("Fisher-Yates Shuffle: [i, j] [%llu, %d]\n", i, j);
        double* swap = temp[i];
        temp[i]      = temp[j];
        temp[j]      = swap;
    }

    for (uint64_t i = 0; i < *sample_size; i++) {
        result[i] = temp[i];
    }

    free(temp);
    return result;
}

static void* build_trees_thread(void* arg)
{
    thread_param* param = (thread_param*)arg;
    uint32_t seed       = param->forest->random_state;
    uint64_t n_samples  = param->data->dimensions[0];
    uint64_t n_features = param->data->dimensions[1];

    printf("thread[%lu] build trees. data-shape(%llu, %llu)\n", (unsigned long)pthread_self(), n_samples, n_features);
    srand(seed);
    for (int i = param->start_tree; i < param->end_tree; i++) {
        // sampling with/without replacement
        uint64_t sample_size = param->forest->num_samples;
        double** subsample   = ndarray_sample_without_replacement(param->data, &sample_size);
        if (subsample == NULL) {
            printf("Memory allocation failed.\n");
            return NULL;
        }

        param->forest->trees[i] = create_node(subsample, n_features, 0,
                                              param->forest->num_samples, 0,
                                              param->forest->max_depth);
        free(subsample);
    }
    return NULL;
}

// static void* build_trees_thread(void* arg)
// {
//     thread_param* param = (thread_param*)arg;
//     for (int i = param->start_tree; i < param->end_tree; i++) {
//         data_point* subsample = malloc(param->forest->subsample_size * sizeof(data_point));
//         CHECK_PTR(subsample);

//         for (int j = 0; j < param->forest->subsample_size; j++) {
//             int idx      = rand() % param->forest->data_size;
//             subsample[j] = param->data[idx];
//         }

//         param->forest->trees[i] = create_node(subsample, 0,
//                                               param->forest->subsample_size, 0,
//                                               param->forest->max_depth);
//         free(subsample);
//     }
//     return NULL;
// }

isolation_forest* iforest_init(int num_trees, int num_samples, int num_features,
                               int num_threads, double contamination, uint32_t random_state)
{
    isolation_forest* forest = calloc(1, sizeof(isolation_forest));
    if (forest == NULL) {
        return NULL;
    }

    forest->num_trees     = num_trees;
    forest->num_samples   = num_samples;
    forest->max_depth     = (int)(ceil(log2(num_samples > 2 ? num_samples : 2))) + 2;
    forest->num_threads   = num_threads;
    forest->num_features  = num_features;
    forest->contamination = contamination;
    forest->random_state  = random_state;

    forest->trees = malloc(num_trees * sizeof(itree_node*));
    if (forest->trees == NULL) {
        forest->num_trees = 0;
    }

    return forest;
}

void iforest_train(isolation_forest* forest, ndarray_t* data)
{
    int num_threads = (forest->num_threads > 0) ? ((forest->num_threads < forest->num_trees) ? forest->num_threads : forest->num_trees) : 1;
    pthread_t threads[num_threads];
    thread_param params[num_threads];
    int trees_per_thread = forest->num_trees / num_threads;

    for (int i = 0; i < num_threads; i++) {
        params[i].forest     = forest;
        params[i].data       = data;
        params[i].start_tree = i * trees_per_thread;
        params[i].end_tree   = (i == num_threads - 1) ? forest->num_trees : (i + 1) * trees_per_thread;
        pthread_create(&threads[i], NULL, build_trees_thread, &params[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

double iforest_score(isolation_forest* forest, double* x)
{
    double avg_path = 0.0;
    for (int i = 0; i < forest->num_trees; i++) {
        int len = itree_get_path_len(forest->trees[i], x);
        avg_path += len;
        // printf("[%d] path len[%d] total-len[%.1f]\n", i, len, avg_path);
    }
    avg_path /= forest->num_trees;
    // printf("Average path length: %.6f, num-trees: %d, Cn: %.6f ret: %.6f \n", avg_path, forest->num_trees, C(forest->num_samples), pow(2, -avg_path / C(forest->num_samples)));
    return pow(2, -avg_path / C(forest->num_samples));
}

void iforest_free(isolation_forest* forest)
{
    for (int i = 0; i < forest->num_trees; i++) {
        if (forest->trees[i]) free_tree(forest->trees[i]);
    }
    free(forest->trees);
    free(forest);
}