#ifndef ISOLATION_FOREST_H
#define ISOLATION_FOREST_H

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ndarray.h"

// Anomaly score constant from original paper: C(n) = H(n-1) + 5.77
#define C(n) (2 * (log((n)-1) + 0.5772156649) - (2 * ((n)-1) / (n)))

// Data point structure
typedef struct {
    double* features;   // Array of feature values
    int num_features;   // Number of features
} data_point;

typedef struct itree_node itree_node;

typedef struct isolation_forest isolation_forest;

isolation_forest* iforest_init(int num_trees, int num_samples, int num_features,
                               int num_threads, double contamination, uint32_t random_state);

void iforest_train(isolation_forest* forest, ndarray_t* data);

// get anomaly score for a data point
double iforest_score(isolation_forest* forest, double* x);

void iforest_free(isolation_forest* forest);

#endif // ISOLATION_FOREST_H