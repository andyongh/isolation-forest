#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Error handling macros
#define CHECK_PTR(ptr)                                                       \
    if (!(ptr)) {                                                            \
        fprintf(stderr, "Allocation failed at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                  \
    }

// Anomaly score constant from original paper
#define C(n) (2 * (log(n - 1) + 0.5772156649) - (2 * (n - 1) / n))

// Data structure for a single data point
typedef struct {
    double* features;  // Array of feature values
    int num_features;  // Number of features
} data_point;

// Node structure for isolation tree
typedef struct itree_node {
    int split_feature;         // Index of feature used for splitting
    double split_value;        // Threshold value for splitting
    struct itree_node* left;   // Left subtree
    struct itree_node* right;  // Right subtree
    int sample_size;           // Number of samples in node
} itree_node;

// Main isolation forest structure
typedef struct {
    itree_node** trees;  // Array of tree pointers
    int num_trees;       // Total number of trees
    int subsample_size;  // Subsampling size per tree
    int max_depth;       // Maximum tree depth
    int num_threads;     // Number of parallel threads
    int num_features;    // Feature dimension
    int data_size;       // Total dataset size
} isolation_forest;

// Thread parameters structure
typedef struct {
    isolation_forest* forest;  // Forest reference
    data_point* data;          // Dataset pointer
    int start_tree;            // Starting tree index
    int end_tree;              // Ending tree index
} thread_param;

/* --------------------- Core Algorithm Functions --------------------- */

itree_node* create_node(data_point* data, int start, int end,
                        int depth, int max_depth)
{
    itree_node* node = (itree_node*)malloc(sizeof(itree_node));
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

    // Find feature range
    for (int i = start + 1; i < end; i++) {
        double val = data[i].features[feat_idx];
        if (val < min) min = val;
        if (val > max) max = val;
    }

    // Generate split value
    double split_val = min + (max - min) * (rand() / (double)RAND_MAX);

    // Partition data
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

double path_length(itree_node* node, data_point* x)
{
    int len             = 0;
    itree_node* current = node;
    if (node->split_feature == -1) {
        return 0;
    }
    while (current != NULL) {
        if (x->features[current->split_feature] < current->split_value) {
            current = current->left;
        } else {
            current = current->right;
        }
        len++;
    }
    return len;
}

void free_tree(itree_node* node)
{
    if (node) {
        free_tree(node->left);
        free_tree(node->right);
        free(node);
    }
}

/* --------------------- Multi-threaded Training --------------------- */

void* build_trees_thread(void* arg)
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
                                              param->forest->subsample_size, 0, param->forest->max_depth);
        free(subsample);
    }
    return NULL;
}

isolation_forest* iforest_init(int num_trees, int subsample_size,
                               int max_depth, int num_threads,
                               int num_features, int data_size)
{
    isolation_forest* forest = malloc(sizeof(isolation_forest));
    CHECK_PTR(forest);

    forest->num_trees      = num_trees;
    forest->subsample_size = subsample_size;
    forest->max_depth      = max_depth;
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
        avg_path += path_length(forest->trees[i], x);
    }
    avg_path /= forest->num_trees;
    printf("Average path length: %.6f, Cn: %.6f ret: %.6f \n", avg_path, C(forest->), pow(2, -avg_path / C(forest->subsample_size)));
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

/* --------------------- CSV Handling --------------------- */

data_point* load_csv(const char* filename, int* num_samples, int* num_features)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[1024];
    *num_samples   = 0;
    *num_features  = 0;
    int has_header = 1;

    // First pass: count samples and features
    while (fgets(line, sizeof(line), file)) {
        if (*num_features == 0) {
            char* token = strtok(line, ",");
            while (token) {
                (*num_features)++;
                token = strtok(NULL, ",");
            }
        }
        (*num_samples)++;
    }

    if (has_header) (*num_samples)--;
    if (*num_samples <= 0 || *num_features <= 0) {
        fprintf(stderr, "Invalid CSV format\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory
    data_point* data = malloc(*num_samples * sizeof(data_point));
    CHECK_PTR(data);
    for (int i = 0; i < *num_samples; i++) {
        data[i].features = malloc(*num_features * sizeof(double));
        CHECK_PTR(data[i].features);
    }

    // Second pass: read data
    rewind(file);
    if (has_header) fgets(line, sizeof(line), file);

    int idx = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        for (int j = 0; j < *num_features; j++) {
            if (!token) {
                fprintf(stderr, "Missing feature at row %d\n", idx + 1);
                exit(EXIT_FAILURE);
            }
            data[idx].features[j] = atof(token);
            token                 = strtok(NULL, ",");
        }
        data[idx].num_features = *num_features;
        idx++;
    }

    fclose(file);
    return data;
}

/* --------------------- Main Program --------------------- */

int main()
{
    srand(time(NULL));

    if (access("test_data.csv", F_OK) == -1) {
        fprintf(stderr, "Error: test_data.csv not found. Run 'make generate_data' first.\n");
        exit(EXIT_FAILURE);
    }

    int num_samples, num_features;
    data_point* data = load_csv("test_data.csv", &num_samples, &num_features);
    printf("Loaded %d samples with %d features\n", num_samples, num_features);

    isolation_forest* forest = iforest_init(
        /* num_trees */ 100,
        /* subsample_size */ 256,
        /* max_depth */ 10,
        /* num_threads */ 4,
        num_features,
        num_samples);
    iforest_train(forest, data);

    // Save scores for comparison
    FILE* output = fopen("c_scores.txt", "w");
    for (int i = 0; i < num_samples; i++) {
        double score = iforest_predict(forest, &data[i]);
        printf("Score %d: %.6f\n", i, score);
        fprintf(output, "%.6f\n", score);
    }
    fclose(output);

    iforest_free(forest);
    for (int i = 0; i < num_samples; i++) {
        free(data[i].features);
    }
    free(data);
    return 0;
}