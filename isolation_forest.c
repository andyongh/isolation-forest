#include "isolation_forest.h"

// Isolation tree node definition
struct itree_node {
    int split_feature;         // Index of feature used for splitting
    double split_value;        // Threshold value for splitting
    struct itree_node* left;   // Left subtree
    struct itree_node* right;  // Right subtree
    int sample_size;           // Number of samples in node
};

// Isolation forest definition
struct isolation_forest {
    itree_node** trees;  // Array of tree pointers
    int num_trees;       // Total number of trees
    int subsample_size;  // Subsampling size per tree
    int max_depth;       // Maximum tree depth
    int num_threads;     // Number of parallel threads
    int num_features;    // Feature dimension
    int data_size;       // Total dataset size
};

// Thread parameters structure
typedef struct {
    isolation_forest* forest;
    data_point* data;
    int start_tree;
    int end_tree;
} thread_param;

/* --------------------- Internal Functions --------------------- */

// Recursively create tree node
static itree_node* create_node(data_point* data, int start, int end, int depth, int max_depth)
{
    itree_node* node = malloc(sizeof(itree_node));
    CHECK_PTR(node);
    node->left = node->right = NULL;

    // Termination conditions
    if (depth >= max_depth || end - start <= 1) {
        node->split_feature = -1;
        node->sample_size   = end - start;
        return node;
    }

    // Random feature selection
    int feat_idx = rand() % data[start].num_features;
    double min   = data[start].features[feat_idx];
    double max   = min;

    // Calculate feature range
    for (int i = start + 1; i < end; i++) {
        double val = data[i].features[feat_idx];
        if (val < min) min = val;
        if (val > max) max = val;
    }

    // Generate split value and partition data
    double split_val = min + (max - min) * (rand() / (double)RAND_MAX);
    int pivot = start;
    for (int i = start; i < end; i++) {
        if (data[i].features[feat_idx] < split_val) {
            data_point tmp = data[pivot];
            data[pivot]    = data[i];
            data[i]        = tmp;
            pivot++;
        }
    }

    // Build subtrees recursively
    node->split_feature = feat_idx;
    node->split_value   = split_val;
    node->sample_size   = end - start;
    node->left          = create_node(data, start, pivot, depth + 1, max_depth);
    node->right         = create_node(data, pivot, end, depth + 1, max_depth);
    return node;
}

// Calculate path length to isolate data point
static int itree_get_path_len(itree_node* node, data_point* x)
{
    int len             = 0;
    itree_node* current = node;
    while (current != NULL && current->split_feature != -1) {
        if (x->features[current->split_feature] < current->split_value) {
            current = current->left;
        } else {
            current = current->right;
        }
        len++;
    }
    return len;
}

// Free tree memory
static void free_tree(itree_node* node)
{
    if (node) {
        free_tree(node->left);
        free_tree(node->right);
        free(node);
    }
}

// Thread function to build trees
static void* build_trees_thread(void* arg)
{
    thread_param* param = (thread_param*)arg;
    for (int i = param->start_tree; i < param->end_tree; i++) {
        data_point* subsample = malloc(param->forest->subsample_size * sizeof(data_point));
        CHECK_PTR(subsample);

        for (int j = 0; j < param->forest->subsample_size; j++) {
            int idx      = rand() % param->forest->data_size;
            subsample[j] = param->data[idx];
        }

        param->forest->trees[i] = create_node(subsample, 0,
                                              param->forest->subsample_size, 0,
                                              param->forest->max_depth);
        free(subsample);
    }
    return NULL;
}

/* --------------------- Public Functions --------------------- */

isolation_forest* iforest_init(int num_trees, int subsample_size, int num_threads,
                               int num_features, int data_size)
{
    isolation_forest* forest = malloc(sizeof(isolation_forest));
    CHECK_PTR(forest);

    forest->num_trees      = num_trees;
    forest->subsample_size = subsample_size;
    forest->max_depth      = (int)(ceil(log2(subsample_size > 2 ? subsample_size : 2))) + 2;
    forest->num_threads    = num_threads;
    forest->num_features   = num_features;
    forest->data_size      = data_size;

    forest->trees = malloc(num_trees * sizeof(itree_node*));
    CHECK_PTR(forest->trees);

    return forest;
}

void iforest_train(isolation_forest* forest, data_point* data)
{
    pthread_t threads[forest->num_threads];
    thread_param params[forest->num_threads];
    int trees_per_thread = forest->num_trees / forest->num_threads;

    for (int i = 0; i < forest->num_threads; i++) {
        params[i].forest     = forest;
        params[i].data       = data;
        params[i].start_tree = i * trees_per_thread;
        params[i].end_tree   = (i == forest->num_threads - 1) ? forest->num_trees : (i + 1) * trees_per_thread;
        pthread_create(&threads[i], NULL, build_trees_thread, &params[i]);
    }

    for (int i = 0; i < forest->num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

double iforest_predict(isolation_forest* forest, data_point* x)
{
    double avg_path = 0.0;
    for (int i = 0; i < forest->num_trees; i++) {
        int len = itree_get_path_len(forest->trees[i], x);
        avg_path += len;
        // printf("[%d] path len[%d] total-len[%.1f]\n", i, len, avg_path);
    }
    avg_path /= forest->num_trees;
    printf("Average path length: %.6f, num-trees: %d, Cn: %.6f ret: %.6f \n", avg_path, forest->num_trees, C(forest->subsample_size), pow(2, -avg_path / C(forest->subsample_size)));
    return pow(2, -avg_path / C(forest->subsample_size));
}

void iforest_free(isolation_forest* forest)
{
    for (int i = 0; i < forest->num_trees; i++) {
        if (forest->trees[i]) free_tree(forest->trees[i]);
    }
    free(forest->trees);
    free(forest);
}