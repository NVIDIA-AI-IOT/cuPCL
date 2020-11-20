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
    Input:
        cloud_in: data pointer for points cloud
        nCount: count of points in cloud_in
    Output:
        Index: data pointer which has the index of points in a plane from input
        modelCoefficients: data pointer which has the group of coefficients of the plane
    */
    int set(segParam_t param);
    void segment(float *cloud_in, int nCount,
            int *index, float *modelCoefficients);
private:
    void *m_handle = NULL;
};


