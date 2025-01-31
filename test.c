#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "isolation_forest.h"

// Load CSV data for testing
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
    data_point* data = malloc(*num_samples * sizeof(data_point));
    CHECK_PTR(data);
    for (int i = 0; i < *num_samples; i++) {
        data[i].features = malloc(*num_features * sizeof(double));
        CHECK_PTR(data[i].features);
    }

    // Second pass: read data
    if (has_header) fgets(line, sizeof(line), file);
    int idx = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        for (int j = 0; j < *num_features; j++) {
            data[idx].features[j] = atof(token);
            token                 = strtok(NULL, ",");
        }
        data[idx].num_features = *num_features;
        idx++;
    }

    fclose(file);
    return data;
}

int main()
{
    srand(time(NULL));

    // Check data file
    if (access("test_data.csv", F_OK) == -1) {
        fprintf(stderr, "Error: test_data.csv not found. Run 'make generate_data' first.\n");
        exit(EXIT_FAILURE);
    }

    // Load data
    int num_samples, num_features;
    data_point* data = load_csv("test_data.csv", &num_samples, &num_features);
    printf("Loaded %d samples with %d features\n", num_samples, num_features);

    // Initialize and train forest
    isolation_forest* forest = iforest_init(100, 256, 4, num_features, num_samples);
    iforest_train(forest, data);

    // Predict and save results
    FILE* output = fopen("c_scores.txt", "w");
    for (int i = 0; i < num_samples; i++) {
        double score = iforest_predict(forest, &data[i]);
        fprintf(output, "%.6f\n", score);
        printf("Score %d: %.6f\n", i, score);
    }
    fclose(output);

    // Cleanup memory
    iforest_free(forest);
    for (int i = 0; i < num_samples; i++) {
        free(data[i].features);
    }
    free(data);

    return 0;
}