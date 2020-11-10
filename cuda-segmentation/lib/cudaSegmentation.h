#pragma once
#include <Eigen/Dense>
#include "cuda_runtime.h"

typedef enum
{
  SACMODEL_PLANE = 0,
  SACMODEL_LINE,
  SACMODEL_CIRCLE2D,
  SACMODEL_CIRCLE3D,
  SACMODEL_SPHERE,
  SACMODEL_CYLINDER,
  SACMODEL_CONE,
  SACMODEL_TORUS,
  SACMODEL_PARALLEL_LINE,
  SACMODEL_PERPENDICULAR_PLANE,
  SACMODEL_PARALLEL_LINES,
  SACMODEL_NORMAL_PLANE,
  SACMODEL_NORMAL_SPHERE,
  SACMODEL_REGISTRATION,
  SACMODEL_REGISTRATION_2D,
  SACMODEL_PARALLEL_PLANE,
  SACMODEL_NORMAL_PARALLEL_PLANE,
  SACMODEL_STICK,
} SacModel;

typedef enum
{
  SAC_RANSAC  = 0,
  SAC_LMEDS   = 1,
  SAC_MSAC    = 2,
  SAC_RRANSAC = 3,
  SAC_RMSAC   = 4,
  SAC_MLESAC  = 5,
  SAC_PROSAC  = 6,
} SacMethod;

typedef struct {
  double distanceThreshold; 
  int maxIterations;
  double probability;
  bool optimizeCoefficients;
} segParam_t;

class cudaSegmentation
{
public:
    //Now Just support: SAC_RANSAC + SACMODEL_PLANE
    cudaSegmentation(int ModelType, int MethodType, cudaStream_t stream = 0);

    ~cudaSegmentation(void);

    /*
    index should >= nCount of maximum inputdata,
    index can be used for multi-inputs, be allocated and freed just at beginning and end
    modelCoefficients can be used for multi-inputs, be allocated and freed just at beginning and end
  */
    int set(segParam_t param);
    void segment(float *cloud_in, int nCount,
            int *index, float *modelCoefficients);
private:
    void *m_handle = NULL;
};


