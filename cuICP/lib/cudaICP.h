/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include "cuda_runtime.h"
#include "Eigen/Core"
#include "Eigen/Dense"

class cudaICP
{
public:
    /*
       nPCountM and nQCountM are the maximum of count for input clouds
       They are used to pre-allocate memory.
    */
    cudaICP(int nPCountM, int nQCountM, cudaStream_t stream = 0);
    ~cudaICP(void);

    /*
    cloud_target = transformation_matrix *cloud_source
    When the Epsilon of transformation_matrix is less than threshold,
    the function will return transformation_matrix.
    Input:
        cloud_source, cloud_target: data pointer for points cloud
        nPCount: the points number of cloud_source
        nQCount: the points number of cloud_target
        Maxiterate: the threshold for iterations
        threshold: When the Epsilon of transformation_matrix is less than
            threshold, the function will return transformation_matrix.
    Output:
        transformation_matrix
    */
    void icp(float *cloud_source, int nPCount,
            float *cloud_target, int nQCount, float relative_mse,
            int Maxiterate, double threshold, float truncted_error,
            void *transformation_matrix,
            cudaStream_t stream = 0);
    void *m_handle = NULL;
};
