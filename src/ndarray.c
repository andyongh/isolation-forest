/*
A NumPy-inspired C library for N-dimensional arrays, supporting machine learning and more.

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

#include "ndarray.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

uint64_t ndarray_size(const ndarray_t* array);

inline static size_t calculate_type_size(char dtype)
{
    if (dtype == 'd') {
        return sizeof(double);
    } else if (dtype == 'f') {
        return sizeof(float);
    } else if (dtype == 'i') {
        return sizeof(uint64_t);
    } else {
        return (size_t)((size_t)dtype < sizeof(uint64_t)) ? (size_t)dtype : 1;
    }
}

static size_t calculate_size(const uint64_t* dims, uint32_t nd)
{
    uint64_t size = 1;
    for (uint32_t i = 0; i < nd; i++) {
        size *= dims[i];
    }
    return size;
}

ndarray_t* ndarray_create(uint64_t* dimensions, uint32_t nd, char dtype)
{
    ndarray_t* array = calloc(1, sizeof(ndarray_t));
    if (!array) return NULL;

    array->nd    = nd;
    array->dtype = dtype;

    // Copy dimensions
    array->dimensions = malloc(nd * sizeof(uint64_t));
    memcpy(array->dimensions, dimensions, nd * sizeof(uint64_t));

    // Calculate strides (assuming row-major order)
    array->strides         = malloc(nd * sizeof(uint64_t));
    array->strides[nd - 1] = calculate_type_size(dtype);
    for (int i = nd - 2; i >= 0; i--) {
        array->strides[i] = array->strides[i + 1] * array->dimensions[i + 1];
    }

    // Allocate data
    uint64_t size    = calculate_size(dimensions, nd);
    size_t type_size = calculate_type_size(dtype);
    array->data      = malloc(size * type_size);

    return array;
}

void ndarray_free(ndarray_t* array)
{
    free(array->data);
    free(array->dimensions);
    free(array->strides);
    free(array);
}

// Create ndarray from CSV file
ndarray_t* ndarray_from_csv(const char* filename, char dtype)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Read dimensions (assuming the first line contains the shape)
    uint64_t n_samples, n_features;
    fscanf(file, "%llu,%llu", &n_samples, &n_features);
    uint64_t dim[2]  = {n_samples, n_features};
    ndarray_t* array = ndarray_create(dim, 2, dtype);
    if (!array) {
        fclose(file);
        return NULL;
    }

    // Read data
    for (uint64_t i = 0; i < n_samples; i++) {
        for (uint64_t j = 0; j < n_features; j++) {
            if (dtype == 'd') {
                fscanf(file, "%lf", (double*)array->data + i * n_features + j);
            } else if (dtype == 'f') {
                fscanf(file, "%f", (float*)array->data + i * n_features + j);
            }
        }
    }

    fclose(file);
    return array;
}

// create an ndarray with random noise based on mean + noise
ndarray_t* ndarray_random_noise(uint64_t n_samples, uint64_t n_features, double mean, double noise_std, char dtype)
{
    uint64_t dim[2]  = {n_samples, n_features};
    ndarray_t* array = ndarray_create(dim, sizeof(dim) / sizeof(dim[0]), dtype);
    if (!array) return NULL;

    srand(time(NULL));
    for (uint64_t i = 0; i < n_samples * n_features; i++) {
        double noise = (double)rand() / RAND_MAX * noise_std * 2 - noise_std;
        if (dtype == 'd') {
            ((double*)array->data)[i] = mean + noise;
        } else if (dtype == 'f') {
            ((float*)array->data)[i] = (float)(mean + noise);
        }
    }

    return array;
}

// create an ndarray with values from a normal distribution
ndarray_t* ndarray_random_normal(uint64_t n_samples, uint64_t n_features, double mean, double std, char dtype)
{
    uint64_t dim[2]  = {n_samples, n_features};
    ndarray_t* array = ndarray_create(dim, 2, dtype);
    if (!array) return NULL;

    srand(time(NULL));
    for (uint64_t i = 0; i < n_samples * n_features; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z  = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);  // Box-Muller transform
        if (dtype == 'd') {
            ((double*)array->data)[i] = mean + std * z;
            printf("%.3f ", ((double*)array->data)[i]);
        } else if (dtype == 'f') {
            ((float*)array->data)[i] = (float)(mean + std * z);
            // printf("%.f ", ((double*)array->data)[i]);
        }
    }

    return array;
}

// Helper functions start ====>
// get data pointer by index tuple. i.e.:(2,3)
void* ndarray_get_point(const ndarray_t* array, uint64_t* idx)
{
    uint64_t offset = 0;
    for (uint32_t i = 0; i < array->nd; i++) {
        uint64_t pos = array->strides[i] * idx[i];
        offset += pos;
        // printf("===>offset:pos, offset, idx[i]\n %llu  %llu %llu\n", pos, offset, idx[i]);
    }
    // printf("===>offset %llu \n", offset);
    // printf("===>stride: %llu, %llu \n", array->strides[0], array->strides[1]);

    return (void*)(((uint8_t*)(array->data)) + offset);
}

// set data pointer by index tuple. i.e.:(2,3)
int ndarray_set_point_d(const ndarray_t* array, uint64_t* idx, double data_point)
{
    uint64_t offset = 0;
    for (uint32_t i = 0; i < array->nd; i++) {
        uint64_t pos = array->strides[i] * idx[i];
        offset += pos;
        // printf("===>offset:pos, offset, idx[i]\n %llu  %llu %llu\n", pos, offset, idx[i]);
    }
    // printf("===>offset %llu \n", offset);
    // printf("===>stride: %llu, %llu \n", array->strides[0], array->strides[1]);
    // printf("==ndarray_set_point==> bf: %.3f \n", *(double*)ndarray_get_point(array, idx));
    // printf("==ndarray_set_point==> bf2: %.3f \n", *(double*)ndarray_get_point(array, idx));
    *((double*)(((uint8_t*)(array->data)) + offset)) = data_point;
    // printf("==ndarray_set_point==> bf: %.3f \n", *(double*)ndarray_get_point(array, idx));

    return 0;
}

int ndarray_set_point_f(const ndarray_t* array, uint64_t* idx, float data_point)
{
    uint64_t offset = 0;
    for (uint32_t i = 0; i < array->nd; i++) {
        uint64_t pos = array->strides[i] * idx[i];
        offset += pos;
    }
    *((float*)(((uint8_t*)(array->data)) + offset)) = data_point;
    return 0;
}

int ndarray_set_point_u64(const ndarray_t* array, uint64_t* idx, uint64_t data_point)
{
    uint64_t offset = 0;
    for (uint32_t i = 0; i < array->nd; i++) {
        uint64_t pos = array->strides[i] * idx[i];
        offset += pos;
    }
    *((uint64_t*)(((uint8_t*)(array->data)) + offset)) = data_point;
    return 0;
}

int ndarray_set_point_u32(const ndarray_t* array, uint64_t* idx, uint32_t data_point)
{
    uint64_t offset = 0;
    for (uint32_t i = 0; i < array->nd; i++) {
        uint64_t pos = array->strides[i] * idx[i];
        offset += pos;
    }
    *((uint32_t*)(((uint8_t*)(array->data)) + offset)) = data_point;
    return 0;
}

int ndarray_set_point(const ndarray_t* array, uint64_t* idx, const void* data_point)
{
    uint64_t offset = 0;
    for (uint32_t i = 0; i < array->nd; i++) {
        uint64_t pos = array->strides[i] * idx[i];
        offset += pos;
    }
    void* dst = (((uint8_t*)(array->data)) + offset);
    memcpy(dst, data_point, calculate_type_size(array->dtype));
    return 0;
}

// Get ndarray Number of Dimensions
uint32_t ndarray_ndim(const ndarray_t* array)
{
    return array->nd;
}

// Get ndarray Shape
const uint64_t* ndarray_shape(const ndarray_t* array)
{
    return array->dimensions;
}

// Get ndarray Shape string
// return tuple string, i.e.: "(2,3,4)"
uint64_t* ndarray_shape_str(const ndarray_t* array)
{
    return array->dimensions;
}

// Get ndarray Data Type
char ndarray_dtype(const ndarray_t* array)
{
    return array->dtype;
}

// Get Total Number of Elements
uint64_t ndarray_size(const ndarray_t* array)
{
    uint64_t size = 1;
    for (uint32_t i = 0; i < array->nd; i++) {
        size *= array->dimensions[i];
    }
    return size;
}

// Helper functions end <====

// Transpose the 2D Array(matrix)
ndarray_t* ndarray_transpose(const ndarray_t* array)
{
    if (array->nd != 2) {
        return NULL;
    }

    uint64_t dim[2]       = {array->dimensions[1], array->dimensions[0]};
    ndarray_t* transposed = ndarray_create(dim, 2, array->dtype);
    if (!transposed) return NULL;

    for (uint64_t i = 0; i < array->dimensions[0]; i++) {
        for (uint64_t j = 0; j < array->dimensions[1]; j++) {
            if (array->dtype == 'd') {
                ((double*)transposed->data)[j * array->dimensions[0] + i] = ((double*)array->data)[i * array->dimensions[1] + j];
            } else if (array->dtype == 'f') {
                ((float*)transposed->data)[j * array->dimensions[0] + i] = ((float*)array->data)[i * array->dimensions[1] + j];
            }
        }
    }

    return transposed;
}

// Arithmetic Operations: Addition
ndarray_t* ndarray_add(ndarray_t* result, const ndarray_t* a, const ndarray_t* b)
{
    if (a->nd != b->nd || a->dtype != b->dtype) {
        return NULL;  // Incompatible dimensions or types
    }

    for (uint32_t i = 0; i < a->nd; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            return NULL;  // Incompatible shapes
        }
    }

    if (!result)
        result = ndarray_create(a->dimensions, a->nd, a->dtype);
    if (!result) return NULL;

    for (uint64_t i = 0; i < a->dimensions[0] * a->dimensions[1]; i++) {
        if (a->dtype == 'd') {
            ((double*)result->data)[i] = ((double*)a->data)[i] + ((double*)b->data)[i];
        } else if (a->dtype == 'f') {
            ((float*)result->data)[i] = ((float*)a->data)[i] + ((float*)b->data)[i];
        }
    }

    return result;
}

// Arithmetic Operations: Subtraction
ndarray_t* ndarray_subtract(ndarray_t* result, const ndarray_t* a, const ndarray_t* b)
{
    if (a->nd != b->nd || a->dtype != b->dtype) {
        return NULL;  // Incompatible dimensions or types
    }

    for (uint32_t i = 0; i < a->nd; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            return NULL;  // Incompatible shapes
        }
    }

    if (!result)
        result = ndarray_create(a->dimensions, a->nd, a->dtype);
    if (!result) return NULL;

    for (uint64_t i = 0; i < a->dimensions[0] * a->dimensions[1]; i++) {
        if (a->dtype == 'd') {
            ((double*)result->data)[i] = ((double*)a->data)[i] - ((double*)b->data)[i];
        } else if (a->dtype == 'f') {
            ((float*)result->data)[i] = ((float*)a->data)[i] - ((float*)b->data)[i];
        }
    }

    return result;
}

// Arithmetic Operations: Dot Product
ndarray_t* ndarray_dot(ndarray_t* result, const ndarray_t* a, const ndarray_t* b)
{
    if (a->nd != 2 || b->nd != 2 || a->dtype != b->dtype || a->dimensions[1] != b->dimensions[0]) {
        return NULL;  // Incompatible dimensions or types
    }

    if (!result)
        result = ndarray_create(a->dimensions, a->nd, a->dtype);
    if (!result) return NULL;

    for (uint64_t i = 0; i < a->dimensions[0]; i++) {
        for (uint64_t j = 0; j < b->dimensions[1]; j++) {
            double sum = 0.0;
            for (uint64_t k = 0; k < a->dimensions[1]; k++) {
                if (a->dtype == 'd') {
                    sum += ((double*)a->data)[i * a->dimensions[1] + k] * ((double*)b->data)[k * b->dimensions[1] + j];
                } else if (a->dtype == 'f') {
                    sum += ((float*)a->data)[i * a->dimensions[1] + k] * ((float*)b->data)[k * b->dimensions[1] + j];
                }
            }
            if (a->dtype == 'd') {
                ((double*)result->data)[i * b->dimensions[1] + j] = sum;
            } else if (a->dtype == 'f') {
                ((float*)result->data)[i * b->dimensions[1] + j] = (float)sum;
            }
        }
    }

    return result;
}

// Broadcasting: Here is a simplified version for 2D arrays
ndarray_t* ndarray_broadcast_add(ndarray_t* result, const ndarray_t* a, const ndarray_t* b)
{
    if (a->nd != 2 || b->nd != 2 || a->dtype != b->dtype) {
        return NULL;  // Incompatible dimensions or types
    }

    uint64_t rows = a->dimensions[0] > b->dimensions[0] ? a->dimensions[0] : b->dimensions[0];
    uint64_t cols = a->dimensions[1] > b->dimensions[1] ? a->dimensions[1] : b->dimensions[1];

    if (!result)
        result = ndarray_create(a->dimensions, a->nd, a->dtype);
    if (!result) return NULL;

    for (uint64_t i = 0; i < rows; i++) {
        for (uint64_t j = 0; j < cols; j++) {
            double val_a = (i < a->dimensions[0] && j < a->dimensions[1]) ? (a->dtype == 'd' ? ((double*)a->data)[i * a->dimensions[1] + j] : ((float*)a->data)[i * a->dimensions[1] + j]) : 0.0;
            double val_b = (i < b->dimensions[0] && j < b->dimensions[1]) ? (b->dtype == 'd' ? ((double*)b->data)[i * b->dimensions[1] + j] : ((float*)b->data)[i * b->dimensions[1] + j]) : 0.0;

            if (a->dtype == 'd') {
                ((double*)result->data)[i * cols + j] = val_a + val_b;
            } else if (a->dtype == 'f') {
                ((float*)result->data)[i * cols + j] = (float)(val_a + val_b);
            }
        }
    }

    return result;
}

// Comparison Operations
ndarray_t* ndarray_compare(ndarray_t* result, const ndarray_t* a, const ndarray_t* b, char op)
{
    if (a->nd != b->nd || a->dtype != b->dtype) {
        return NULL;  // Incompatible dimensions or types
    }

    for (uint32_t i = 0; i < a->nd; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            return NULL;  // Incompatible shapes
        }
    }

    if (!result)
        result = ndarray_create(a->dimensions, a->nd, a->dtype);
    if (!result) return NULL;

    for (uint64_t i = 0; i < a->dimensions[0] * a->dimensions[1]; i++) {
        double val_a = a->dtype == 'd' ? ((double*)a->data)[i] : ((float*)a->data)[i];
        double val_b = b->dtype == 'd' ? ((double*)b->data)[i] : ((float*)b->data)[i];

        switch (op) {
        case '>':
            ((int*)result->data)[i] = val_a > val_b;
            break;
        case '<':
            ((int*)result->data)[i] = val_a < val_b;
            break;
        case '=':
            ((int*)result->data)[i] = val_a == val_b;
            break;
        default:
            ndarray_free(result);
            return NULL;
        }
    }

    return result;
}

// Subsampling Without Replacement
ndarray_t* ndarray_subsample(const ndarray_t* array, uint64_t n_samples)
{
    if (n_samples > array->dimensions[0]) {
        return NULL;  // Cannot sample more than available
    }

    // Create new dimensions array
    uint64_t* new_dims = malloc(array->nd * sizeof(uint64_t));
    if (!new_dims) return NULL;
    new_dims[0] = n_samples;
    for (uint32_t i = 1; i < array->nd; i++) {
        new_dims[i] = array->dimensions[i];
    }

    ndarray_t* subsampled = ndarray_create(new_dims, array->nd, array->dtype);
    free(new_dims);
    if (!subsampled) return NULL;

    srand(time(NULL));
    uint64_t* indices = malloc(array->dimensions[0] * sizeof(uint64_t));
    for (uint64_t i = 0; i < array->dimensions[0]; i++) {
        indices[i] = i;
    }

    // Shuffle indices
    for (uint64_t i = 0; i < array->dimensions[0]; i++) {
        uint64_t j    = rand() % array->dimensions[0];
        uint64_t temp = indices[i];
        indices[i]    = indices[j];
        indices[j]    = temp;
    }

    // Copy data
    for (uint64_t i = 0; i < n_samples; i++) {
        for (uint64_t j = 0; j < array->dimensions[1]; j++) {
            if (array->dtype == 'd') {
                ((double*)subsampled->data)[i * array->dimensions[1] + j] = ((double*)array->data)[indices[i] * array->dimensions[1] + j];
            } else if (array->dtype == 'f') {
                ((float*)subsampled->data)[i * array->dimensions[1] + j] = ((float*)array->data)[indices[i] * array->dimensions[1] + j];
            }
        }
    }

    free(indices);
    return subsampled;
}

// Concatenation
ndarray_t* ndarray_concat(const ndarray_t* a, const ndarray_t* b, uint32_t axis)
{
    // Validate inputs
    if (!a || !b) return NULL;
    if (a->nd != b->nd) return NULL;
    if (axis >= a->nd) return NULL;
    if (a->dtype != b->dtype) return NULL;

    // Check dimension compatibility
    for (uint32_t i = 0; i < a->nd; i++) {
        if (i != axis && a->dimensions[i] != b->dimensions[i]) {
            return NULL;
        }
    }

    // Create new dimensions array
    uint64_t* new_dims = malloc(a->nd * sizeof(uint64_t));
    if (!new_dims) return NULL;

    for (uint32_t i = 0; i < a->nd; i++) {
        new_dims[i] = (i == axis) ? (a->dimensions[i] + b->dimensions[i]) : a->dimensions[i];
    }

    // Create result array
    ndarray_t* result = ndarray_create(new_dims, a->nd, a->dtype);
    if (!result) return NULL;
    free(new_dims);
    if (!result) return NULL;

    // Calculate element count and element size
    uint64_t total_elements = ndarray_size(result);
    size_t type_size        = calculate_type_size(result->dtype);

    // Allocate indices array
    uint64_t* indices = malloc(result->nd * type_size);
    if (!indices) {
        ndarray_free(result);
        return NULL;
    }

    // Copy data element-wise
    for (uint64_t linear_idx = 0; linear_idx < total_elements; linear_idx++) {
        // Calculate multidimensional indices
        uint64_t remainder = linear_idx;
        for (uint32_t i = 0; i < result->nd; i++) {
            indices[i] = remainder / result->strides[i];
            remainder %= result->strides[i];
        }

        // Determine source array and calculate source index
        const ndarray_t* src;
        uint64_t axis_idx = indices[axis];
        uint64_t src_axis;

        if (axis_idx < a->dimensions[axis]) {
            src      = a;
            src_axis = axis_idx;
        } else {
            src      = b;
            src_axis = axis_idx - a->dimensions[axis];
        }

        // Calculate source linear index
        uint64_t src_idx = 0;
        for (uint32_t i = 0; i < src->nd; i++) {
            src_idx += (i == axis) ? src_axis * src->strides[i] : indices[i] * src->strides[i];
        }

        // Copy data
        if (a->dtype == 'd') {
            ((double*)result->data)[linear_idx] = ((double*)src->data)[src_idx];
        } else {
            ((float*)result->data)[linear_idx] = ((float*)src->data)[src_idx];
        }
    }

    free(indices);
    return result;
}
