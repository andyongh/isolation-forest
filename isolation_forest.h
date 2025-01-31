#ifndef ISOLATION_FOREST_H
#define ISOLATION_FOREST_H

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
#define C(n) (2 * (log((n)-1) + 0.5772156649) - (2 * ((n)-1) / (n)))

// Data point structure
typedef struct {
    double* features;   // Array of feature values
    int num_features;   // Number of features
} data_point;

// Opaque isolation tree node
typedef struct itree_node itree_node;

// Opaque isolation forest structure
typedef struct isolation_forest isolation_forest;

// Initialize forest
isolation_forest* iforest_init(int num_trees, int subsample_size, int num_threads,
                              int num_features, int data_size);

// Train forest
void iforest_train(isolation_forest* forest, data_point* data);

// Predict anomaly score for a data point
double iforest_predict(isolation_forest* forest, data_point* x);

// Free forest memory
void iforest_free(isolation_forest* forest);

#endif // ISOLATION_FOREST_H