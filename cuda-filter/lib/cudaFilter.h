#pragma once
#include "cuda_runtime.h"
#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

typedef enum {
    PASSTHROUGH=0,
} FilterType_t;

typedef struct {
    FilterType_t type;
    //type PASSTHROUGH
    int dim;
    float upFilterLimits;
    float downFilterLimits;
    bool limitsNegative;

} FilterParam_t;

class cudaFilter
{
public:
    /*
       nPCountM and nQCountM are the maximum of count for input clouds
       They are used to pre-allocate memory.
    */
    cudaFilter(cudaStream_t stream = 0);
    ~cudaFilter(void);
    int set(FilterParam_t param);
    int filter(void *output, unsigned int *countLeft, void *source, unsigned int nCount);

    void *m_handle = NULL;
};

