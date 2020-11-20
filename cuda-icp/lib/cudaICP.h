/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#pragma once
#include <Eigen/Dense>
#include "cuda_runtime.h"

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
            float *cloud_target, int nQCount,
            int Maxiterate, double threshold,
            Eigen::Matrix4f &transformation_matrix,
            cudaStream_t stream = 0);
    void *m_handle = NULL;
};
