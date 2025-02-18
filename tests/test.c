#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "isolation_forest.h"
#include "ndarray.h"

#define CHECK_PTR(ptr)                                                       \
    if (!(ptr)) {                                                            \
        fprintf(stderr, "Allocation failed at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                  \
    }

// Load CSV data for testing
ndarray_t* load_csv(const char* filename, int* num_samples, int* num_features)
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

    // First pass: count features and samples
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
    rewind(file);

    // Allocate memory
    uint32_t nd = 2;
    // *num_samples     = 200;
    uint64_t dims[2] = {*num_samples, *num_features};
    ndarray_t* data = ndarray_create(dims, nd, 'f');
    CHECK_PTR(data);

    // Second pass: read data
    if (has_header) fgets(line, sizeof(line), file);
    int idx = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        // double* point = &(data->data[idx * data->strides[0]]);
        for (int i = 0; i < *num_features; i++) {
            // point[i]              = atof(token);
            float p               = atof(token);
            token                 = strtok(NULL, ",");
            uint64_t npos[2] = {idx, i};
            ndarray_set_point_f(data, npos, p);
            // printf("idx[%d], point[%d] %1.3f\n", idx, i, point[i]);
        }
        idx++;
    }

    fclose(file);
    return data;
}

int main(int argc, const char *argv[])
{
    const char* file = "./test_data.csv";

    if (argc > 1) {
        file = argv[1];
    }
    if (access(file, F_OK) == -1) {
        fprintf(stderr, "Error: test_data.csv not found. Run 'make generate_data' first.\n");
        exit(EXIT_FAILURE);
    }

    // Load data
    srand(time(NULL));
    int num_samples, num_features;
    ndarray_t* data = load_csv(file, &num_samples, &num_features);
    printf("Loaded %d samples with %d features\n", num_samples, num_features);
    printf("ndarray dimensions[%u] shape(%llu, %llu)\n", data->nd, data->dimensions[0], data->dimensions[1]);
    printf("ndarray stride(%llu, %llu) \n", data->strides[0], data->strides[1]);
    for (int i = 0; i < 10; i++) {
        // double* point = &data->data[i * data->strides[0]];
        printf("ndarray point[%d] ", i);
        for (int j = 0; j < num_features; j++) {
            uint64_t npos[2] = {i, j};
            float p          = *(float *)ndarray_get_point(data, npos);
            printf("%2.3f ", p);
        }
        printf("\n");
    }

    // Initialize and train forest
    isolation_forest* forest = iforest_init(100, 256, num_features, 4, 0, 42);
    iforest_train(forest, data);

    FILE* output = fopen("c_scores.txt", "w");
    for (int i = 0; i < num_samples; i++) {
        printf("infer point[%d] ", i);
        for (int j = 0; j < num_features; j++) {
            uint64_t npos[2] = {i, j};
            float p          = *(float *)ndarray_get_point(data, npos);
            printf("%.3f ", p);
        }
        printf("\n");
        uint64_t npos[2] = {i, 0};
        float* point     = ndarray_get_point(data, npos);
        double score     = iforest_score(forest, point);
        fprintf(output, "%.6f\n", score);
        printf("Score %d: %.6f\n", i, score);
    }
    fclose(output);

    // Cleanup memory
    iforest_free(forest);
    ndarray_free(data);

    return 0;
}