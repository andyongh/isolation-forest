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

#ifndef NDARRAY_H
#define NDARRAY_H

#include <stdint.h>

typedef struct {
    void* data;            // Pointer to array data
    uint64_t* dimensions;  // Array shape
    uint64_t* strides;     // Strides between elements
    uint32_t nd;           // Number of dimensions
    char dtype;            // Data type ('d' for double, 'f' for float, 'i' for integer)
} ndarray_t;

ndarray_t* ndarray_create(uint64_t* dimensions, uint32_t nd, char dtype);
void ndarray_free(ndarray_t* array);

ndarray_t* ndarray_from_csv(const char* filename, char dtype);

ndarray_t* ndarray_random_noise(uint64_t n_samples, uint64_t n_features, double mean, double noise_std, char dtype);
ndarray_t* ndarray_random_normal(uint64_t n_samples, uint64_t n_features, double mean, double std, char dtype);

int ndarray_set_point(const ndarray_t* array, uint64_t* idx, const void* data_point);
int ndarray_set_point_d(const ndarray_t* array, uint64_t* idx, double data_point);
int ndarray_set_point_f(const ndarray_t* array, uint64_t* idx, float data_point);
int ndarray_set_point_u64(const ndarray_t* array, uint64_t* idx, uint64_t data_point);
int ndarray_set_point_u32(const ndarray_t* array, uint64_t* idx, uint32_t data_point);
void* ndarray_get_point(const ndarray_t* array, uint64_t* idx);
uint32_t ndarray_ndim(const ndarray_t* array);
const uint64_t* ndarray_shape(const ndarray_t* array);
char ndarray_dtype(const ndarray_t* array);
uint64_t ndarray_size(const ndarray_t* array);

// Arithmetic operations
ndarray_t* ndarray_add(ndarray_t* result, const ndarray_t* a, const ndarray_t* b);
ndarray_t* ndarray_subtract(ndarray_t* result, const ndarray_t* a, const ndarray_t* b);
ndarray_t* ndarray_dot(ndarray_t* result, const ndarray_t* a, const ndarray_t* b);
ndarray_t* ndarray_broadcast_add(ndarray_t* result, const ndarray_t* a, const ndarray_t* b);

// Comparison
ndarray_t* ndarray_compare(ndarray_t* result, const ndarray_t* a, const ndarray_t* b, char op);

// Transformations
ndarray_t* ndarray_transpose(const ndarray_t* array);
ndarray_t* ndarray_subsample(const ndarray_t* array, uint64_t n_samples);

// Concatenation
ndarray_t* ndarray_concat(const ndarray_t* a, const ndarray_t* b, uint32_t axis);

#endif