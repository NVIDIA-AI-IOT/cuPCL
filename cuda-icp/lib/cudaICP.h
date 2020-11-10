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
        nPCount is the points number of cloud_target
        nQCount is the points number of cloud_target
        Maxiterate is the threshold for iterations
        When the Epsilon of transformation_matrix is less than threshold,
        the function will return transformation_matrix.
    */
    void icp(float *cloud_source, int nPCount,
            float *cloud_target, int nQCount,
            int Maxiterate, double threshold,
            Eigen::Matrix4f &transformation_matrix,
            cudaStream_t stream = 0);
    void *m_handle = NULL;
};
