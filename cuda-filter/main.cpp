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
#include <fstream>
#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/random_sample.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "cuda_runtime.h"
#include "lib/cudaFilter.h"

void Getinfo(void)
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

void testCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  cudaStream_t stream = NULL;
  cudaStreamCreate ( &stream );

  unsigned int nCount = cloudSrc->width * cloudSrc->height;
  float *inputData = (float *)cloudSrc->points.data();

  cloudDst->width  = nCount;
  cloudDst->height = 1;
  cloudDst->resize (cloudDst->width * cloudDst->height);

  float *outputData = (float *)cloudDst->points.data();

  memset(outputData,0,sizeof(float)*4*nCount);

  //std::cout << "\n------------checking CUDA ---------------- "<< std::endl;
  std::cout << "CUDA Loaded "
      << cloudSrc->width*cloudSrc->height
      << " data points from PCD file with the following fields: "
      << pcl::getFieldsList (*cloudSrc)
      << std::endl;

  float *input = NULL;
  cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, input );
  cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  float *output = NULL;
  cudaMallocManaged(&output, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, output );
  cudaMemcpyAsync(output, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  unsigned int countLeft = 0;
  cudaFilter filterTest(stream);
  FilterParam_t setP;
  FilterType_t type;

{
  std::cout << "\n------------checking CUDA PassThrough ---------------- "<< std::endl;

  memset(outputData,0,sizeof(float)*4*nCount);

  FilterType_t type = PASSTHROUGH;

  setP.type = type;
  setP.dim = 2;
  setP.upFilterLimits = 1.0;
  setP.downFilterLimits = 0.0;
  setP.limitsNegative = false;
  filterTest.set(setP);

  cudaDeviceSynchronize();
  t1 = std::chrono::steady_clock::now();
  filterTest.filter(output, &countLeft, input, nCount);
  checkCudaErrors(cudaMemcpyAsync(outputData, output, sizeof(float) * 4 * countLeft, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaDeviceSynchronize());
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA PassThrough by Time: " << time_span.count() << " ms."<< std::endl;
  std::cout <<"Points selected: "<< countLeft<< std::endl;
}

}

void testPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst)
{
  //std::cout << "\n\n------------checking PCL ---------------- "<< std::endl;
  std::cout << "\n\nPCL(CPU) Loaded "
      << cloudSrc->width*cloudSrc->height
      << " data points from PCD file with the following fields: "
      << pcl::getFieldsList (*cloudSrc)
      << std::endl;
  
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  int nCount = cloudSrc->width * cloudSrc->height;
  float *outputData = (float *)cloudDst->points.data();
{
  std::cout << "\n------------checking PCL(CPU) PassThrough ---------------- "<< std::endl;

  memset(outputData,0,sizeof(float)*4*nCount);

  // Create the filtering object
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloudSrc);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.setFilterLimitsNegative (false);

  t1 = std::chrono::steady_clock::now();
  pass.filter (*cloudDst);
  t2 = std::chrono::steady_clock::now();

  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "PCL(CPU) PassThrough by Time: " << time_span.count() << " ms."<< std::endl;
  /*
  std::cout << "PointCloud before filtering: " << cloudSrc->width * cloudSrc->height 
   << " data points (" << pcl::getFieldsList (*cloudSrc) << ")." << std::endl;
  */
  std::cout << "PointCloud after filtering: " << cloudDst->width * cloudDst->height 
     << " data points (" << pcl::getFieldsList (*cloudDst) << ")." << std::endl;
}

}

int main(int argc, const char **argv)
{
  std::string file = "./sample.pcd";
  if(argc > 1) file = (argv[1]);

  Getinfo();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(file.c_str(), *cloudSrc )== -1)
  {
    std::cout << "Error:can not open the file: "<< file.c_str() << std::endl;
    return(-1);
  }

  testCUDA(cloudSrc, cloudDst);
  testPCL(cloudSrc, cloudDst);

  return 0;
}
